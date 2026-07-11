#!/usr/bin/env bash
# Run the container exactly like the judging harness: 4 GB RAM, 2 vCPU,
# /input and /output mounts, env from .env.
#
# Usage: eval/run_local.sh <IMAGE> [INPUT_DIR] [--accuracy]
#   INPUT_DIR    directory containing tasks.json (default: test_input)
#   --accuracy   lift the deadlines (TOTAL_DEADLINE_S=7200, PER_TASK_CAP_S=300)
#                to measure pure accuracy on the big test set, no time pressure
#   PLATFORM=... optionally force a platform (omit to use the image's native one,
#                e.g. for the :dev-arm build on Apple Silicon)
set -euo pipefail
cd "$(dirname "$0")/.."

IMAGE="${1:?usage: eval/run_local.sh <image> [input_dir] [--accuracy]}"
shift
INPUT_DIR="test_input"
EXTRA_ENV=()
for arg in "$@"; do
  case "$arg" in
    --accuracy) EXTRA_ENV+=(-e TOTAL_DEADLINE_S=7200 -e PER_TASK_CAP_S=300) ;;
    *) INPUT_DIR="$arg" ;;
  esac
done
[ -f "$INPUT_DIR/tasks.json" ] || { echo "no $INPUT_DIR/tasks.json (run: make testset)"; exit 1; }

PLATFORM_ARGS=()
[ -n "${PLATFORM:-}" ] && PLATFORM_ARGS=(--platform "$PLATFORM")
ENV_ARGS=()
[ -f .env ] && ENV_ARGS=(--env-file .env)

mkdir -p test_output
rm -f test_output/results.json

echo "running $IMAGE on $INPUT_DIR with 4g/2cpu limits ${EXTRA_ENV[*]:-}..."
START=$(date +%s)
set +e
docker run --rm ${PLATFORM_ARGS[@]+"${PLATFORM_ARGS[@]}"} --memory=4g --cpus=2 \
  -v "$PWD/$INPUT_DIR:/input:ro" -v "$PWD/test_output:/output" \
  ${ENV_ARGS[@]+"${ENV_ARGS[@]}"} ${EXTRA_ENV[@]+"${EXTRA_ENV[@]}"} "$IMAGE"
CODE=$?
set -e
END=$(date +%s)
WALL=$((END - START))

echo ""
echo "exit code: $CODE | wall time: ${WALL}s"
if [ "$CODE" -ne 0 ]; then
  echo "container failed; results validation skipped" >&2
  exit "$CODE"
fi
if [ ! -f test_output/results.json ]; then
  echo "container exited 0 but did not write test_output/results.json" >&2
  exit 1
fi
N_IN=$(python3 -c "import json;print(len(json.load(open('$INPUT_DIR/tasks.json'))))")
N_OUT=$(python3 -c "import json;print(len(json.load(open('test_output/results.json'))))")
echo "tasks: $N_IN in / $N_OUT answered | avg $(python3 -c "print(f'{$WALL/max(1,$N_IN):.1f}')")s per task"
python3 - <<'EOF'
import json
results = json.load(open("test_output/results.json"))
empty = [r["task_id"] for r in results if not str(r.get("answer", "")).strip()]
print("empty answers:", empty or "none")
print("results.json is valid JSON with", len(results), "entries")
EOF
echo "diagnostics: see the 'done:' line in the container stderr above"
