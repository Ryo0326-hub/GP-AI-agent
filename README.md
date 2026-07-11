# GP-AI-Agent — AMD Hackathon Track 1

Submission image: `rkitano/gp-agent:v10-remote`

- Published digest: `sha256:b586ef5ac3d8ec71d4ffa463a6d5e16a992c859fb27fcc02a4c0ff2444c52f06`
- Platform: `linux/amd64`
- Acceptance rehearsal: 18/19 (94.7%), 49 seconds, 19/19 non-empty answers

## Demo application

The Vercel-ready Next.js showcase lives in [`demo/`](demo/). It includes an
interactive task console, architecture walkthrough, category coverage, and
benchmark evidence. It works without secrets in labeled simulation mode; add
the Fireworks variables from `demo/.env.example` to enable live answers.

```bash
cd demo
npm install
npm run dev
```

For Vercel, import this repository and set **Root Directory** to `demo`. The
framework preset, install command, and build command are detected automatically.

Token-efficient general-purpose agent. Local-first: a Qwen2.5-3B-Instruct
Q4_K_M GGUF baked into the image answers everything at **zero Fireworks
tokens**; deterministic verifiers (arithmetic evaluator, sandboxed code tests,
logic brute-force, summary-constraint checks) decide when an answer is trusted.
Only unverified answers escalate to an accuracy-safe model selected from `ALLOWED_MODELS`
via `FIREWORKS_BASE_URL`.

## Quick start

```bash
export IMAGE=<dockerhub-user>/gp-agent:latest
make test                # unit tests (no model needed)
make build-dev           # native arm64 dev image (:dev-arm) for Apple Silicon iteration
make build               # linux/amd64 submission image, GGUF downloaded at build time
make build-remote        # lightweight API-first fallback (no bundled GGUF)
make testset             # test_input/ (80 tasks) + test_input_19/ (rehearsal) + expected.json
make check-models        # probe every ALLOWED_MODELS id via your .env (5 tokens each)
make judge               # deterministic + LLM-judge scoring
make build-push          # final linux/amd64 buildx --push
```

Runs (dev):

```bash
# timing rehearsal: 19 tasks, real deadlines, native arm image
PLATFORM= bash eval/run_local.sh gp-agent:dev-arm test_input_19
# accuracy pass: 80 tasks, deadlines lifted (TOTAL_DEADLINE_S=7200, PER_TASK_CAP_S=300)
bash eval/run_local.sh gp-agent:dev-arm test_input --accuracy
python3 eval/judge.py                     # scores test_input vs test_output
python3 eval/judge.py test_input_19       # scores the rehearsal run
```

`.env` (dev only, never in the image):

```
FIREWORKS_API_KEY=...
FIREWORKS_BASE_URL=https://api.fireworks.ai/inference/v1
ALLOWED_MODELS=accounts/fireworks/models/llama-v3p1-8b-instruct
```

## Rule-compliance checklist

| Rule | Where satisfied |
|---|---|
| Reads `/input/tasks.json` array of `{task_id, prompt}` | `app/main.py` `load_tasks` (with retry) |
| Writes `/output/results.json` `{task_id, answer}` per task, valid JSON, exit 0 | `app/main.py`: initial `[]` write, incremental `write_partial` (atomic temp+rename in `utils.atomic_write_json`), final validation pass, `sys.exit(0)`; fatal-error handler still writes valid JSON |
| linux/amd64, ≤10 GB compressed | `Dockerfile` (python:3.11-slim + ~2 GB GGUF ≈ 2.5 GB); `make build/push` pin `--platform linux/amd64` |
| 4 GB RAM / 2 vCPU, no GPU | 3B Q4_K_M ≈ 2 GB resident; `llama-cpp-python` CPU build, `n_threads=2` (`app/local_model.py`) |
| Container ready < 60 s | Model load is the only startup cost (seconds from local disk); benchmark is ~30 tokens |
| Total runtime < 10 min | 9-min internal deadline (`TOTAL_DEADLINE_S=540`); behind-pace ⇒ concurrent bulk escalation (`bulk_escalate`) |
| < 30 s per task | `PER_TASK_CAP_S=25` end-to-end; `LOCAL_SOLVE_CAP_S=18` reserves time for escalation, and Fireworks retries share the remaining deadline |
| English answers | English prompts/templates throughout (`app/prompts.py`) |
| `FIREWORKS_API_KEY` / `FIREWORKS_BASE_URL` only, from env | `app/fireworks.py` reads env at runtime; single call path `BASE_URL + /chat/completions` |
| Only `ALLOWED_MODELS` | `Fireworks._pick_model` selects from the env list only (prefers direct-answer GPT-OSS when present, then compact models, fallback to first) |
| No hardcoded/cached answers | Nothing keyed on prompt text; classifier routes by surface form only, all answers generated |
| No secrets in image | `.dockerignore` excludes `.env`, `eval/`, `tests/` |

## Tuning knobs (accuracy ↔ tokens)

- **Model choice** (`--build-arg MODEL_URL=...`): 3B = better accuracy, ~2× slower;
  1.5B if the startup benchmark shows < ~6 tok/s on the judging VM.
- **`MAX_TOKENS` / `ESCALATION_MAX_TOKENS`** (`app/prompts.py`): lower = faster +
  cheaper escalations, but risks truncated answers the judge marks wrong.
- **`ESCALATE_UNVERIFIED_LOGIC`** (env, default `1`): `0` trusts local logic answers
  that end in `Answer:` — saves tokens, risks the accuracy gate.
- **Verifier strictness**: `looks_confident` min length, NER line regex, summary
  constraint parsing (`app/verify.py`) — stricter = more escalations = more tokens.
- **`PER_TASK_MIN_S` / `TOTAL_DEADLINE_S` / `PER_TASK_CAP_S` / `LOCAL_SOLVE_CAP_S`** (env, `app/main.py`):
  the pacer never gives a task less than `PER_TASK_MIN_S` (default 20 s) — if the
  floor × remaining tasks won't fit the deadline, it bulk-escalates instead of
  truncating every answer. Deadlines are env-overridable for dev accuracy runs.
- **`MIN_LOCAL_TOK_S`** (env, default `4`): startup benchmark circuit breaker;
  slower CPUs route directly to Fireworks so decode latency cannot cause fallbacks.

The `done:` stderr line reports truthful diagnostics: per-path counts
(local-verified / local-unverified / escalated / fallback), Fireworks calls
attempted vs succeeded, total tokens, generation cutoffs, and the category
histogram over **all** tasks. Escalation errors log the HTTP response body;
404s are non-retryable, mark the model known-bad for the run, and fall back to
the next-smallest allowed model.

Note: Qwen2.5-3B-Instruct is under the Qwen Research License; 1.5B/7B are
Apache-2.0. Swap the `MODEL_URL` build arg if licensing matters for your entry.
