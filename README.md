# GP-AI-Agent — AMD Hackathon Track 1

GP-AI-Agent is a local-first, token-efficient general-purpose agent. A compact
learned router predicts whether the bundled Qwen pipeline should answer a task
locally or send it to an allowlisted Fireworks model. Local inference and local
routing use zero Fireworks tokens; deterministic checks remain a second safety
gate before an unverified answer is trusted.

The routing approach is based on the AMD query-router
[tutorial](https://lablab.ai/ai-tutorials/fine-tune-llm-query-router-amd) and
[reference repository](https://github.com/Stephen-Kimoi/fine-tune-llm-query-router-amd),
adapted for this agent's measured local outcomes and the Track 1 container
constraints.

## Benchmark status

The last submitted image, `rkitano/gp-agent:v10-remote`, scored **18/19
(94.7%)** and used **9,685 Fireworks tokens**. Those numbers describe the
previous API-first submission, not the new learned-router candidate.

Measured components for the new candidate:

- Qwen2.5-1.5B-Instruct Q4_K_M produced approximately **4.9–5.8 tok/s** in
  isolated 2-vCPU labeling runs. The 3B model produced approximately **2.2
  tok/s**, below the runtime's 4 tok/s circuit-breaker, so the Dockerfile now
  uses the pinned 1.5B model.
- On the 20-task hard escalation set, DeepSeek V4 Flash with the category
  system prompt and `reasoning_effort: "none"` scored **19/20 (95%)** using
  **3,365 tokens**. The same model with the raw prompt scored **17/20 (85%)**
  using **4,464 tokens**: the category-aware prompt improved accuracy by 10
  percentage points while reducing tokens by 24.6%.

The escalation measurements are preserved in
[`data/escalation_prompt_style_report.json`](data/escalation_prompt_style_report.json)
and [`data/escalation_model_report.json`](data/escalation_model_report.json).

The v11 compact-router profile is revision `compact-80a9ce84c395`: holdout
accuracy is 77.55%, failure recall is 100%, slow-failure planned recall is
100%, and unsafe-local count is 0. The arm64 rehearsal answered 19/19 in
70.5s; the 80-task pass answered all 80 in 207.4s with 45 successful remote
calls and no fallback failures. The deterministic judge scored 59/60 (98.3%);
the remaining hosted-judge items require Fireworks network access.

The optimized candidate is published as
[`rkitano/gp-agent:v11`](https://hub.docker.com/r/rkitano/gp-agent/tags?name=v11)
(`linux/amd64`, digest
`sha256:2020082323bbc2c1d22d641d14bb66760d2acd1a2927f4c71333454df9d46a69`).
The walkthrough is live at <https://gp-ai-agent-1.vercel.app/> and displays the
previous baseline beside the v11 projected accuracy and expected token spend.

## Architecture

1. **Empirical labels.** `eval/label_local.py` runs the complete local pipeline
   against the 360-task ground-truthed training set under a 2-vCPU, 18-second
   local budget. Each example is labeled `local_ok` or `escalate` from the
   observed answer, verifier result, and expected answer.
2. **Compact learned router.** `router/train_compact_router.py` fits a
   deterministic hashed unigram/bigram logistic classifier. Training,
   calibration, and one-shot test groups are separated by prompt-template
   similarity. The escalation threshold is calibrated separately, while
   category policy and completion-cost estimates come only from the training
   split. No Torch or Transformers runtime is included in the final image.
3. **Artifact parity.** The trainer writes byte-identical JSON profiles for the
   Python runtime and browser demo. `make router-artifact-check` fails closed if
   the profile is pending, labels do not exactly cover `train_data/tasks.json`,
   the held-out metrics are missing, or the two artifacts differ.
4. **Budget-aware local plan.** `app/main.py` scores all tasks before loading
   the GGUF. Planned escalations start concurrently while local tasks are
   ordered by measured completion cost and admitted against the remaining
   wall-clock budget.
5. **Verification safety gate.** Arithmetic is recomputed, generated code is
   tested in a subprocess, supported logic puzzles are brute-forced, and
   summary constraints are checked. A failed or untrusted local answer can
   escalate within the same end-to-end task deadline.
6. **Category-aware escalation.** Every hosted request includes the same
   category-specific system instruction used by the local solver. The default
   build prefers an allowed model whose name contains `deepseek-v4-flash` and
   sends only the allowlisted extra field `{"reasoning_effort":"none"}`.
   Model IDs still come exclusively from the runtime `ALLOWED_MODELS` value.
7. **Graceful degradation.** A broken router falls back to
   verification-gated local solving. A missing GGUF or local speed below 4
   tok/s switches to Fireworks when it is available.

The optional `make train-distilbert` target preserves a 66M-parameter research
comparison with the tutorial. It is not the router shipped by the current
Dockerfile.

## Reproduce and validate

Prerequisites: Python 3, `uv`, Docker with Buildx, and a Fireworks key for
hosted-model measurements and end-to-end rehearsals.

```bash
# One-time local setup and deterministic data
make venv
make models-download
make dataset
make testset

# Measure the chosen local pipeline under its production budget
make label-15b

# Fit the shared runtime/browser profile and validate exact label coverage
make train-router
make router-artifact-check
make router-check

# Re-run unit tests before building
make test
```

`make train-router` deliberately excludes prompts from `test_input/` and
`test_input_19/` before splitting the labels. Both build targets depend on the
artifact check, so a placeholder or stale profile cannot be packaged by
accident.

For local Fireworks measurements, create a root `.env` that is never committed:

```dotenv
FIREWORKS_API_KEY=replace_with_your_key
FIREWORKS_BASE_URL=https://api.fireworks.ai/inference/v1
ALLOWED_MODELS=accounts/fireworks/models/deepseek-v4-flash,accounts/fireworks/models/another_allowed_model
```

Then measure the escalation tier and run the candidate under judge-like limits:

```bash
make pick-model
make build-dev
bash eval/run_local.sh gp-agent:v11-dev-arm test_input_19
python3 eval/judge.py test_input_19
bash eval/run_local.sh gp-agent:v11-dev-arm test_input --accuracy
python3 eval/judge.py
```

After the validation gates pass, build and push the final architecture:

```bash
make build-push IMAGE=rkitano/gp-agent:v11
```

The default build is equivalent to:

```text
HINTS=deepseek-v4-flash,deepseek-v4-pro
EXTRA_BODY={"reasoning_effort":"none"}
```

The submission image targets amd64. On an ARM64 development machine, pull it
with explicit emulation; the hackathon evaluator can pull the tag normally:

```bash
docker pull --platform linux/amd64 rkitano/gp-agent:v11
```

`HINTS` is a name pattern matched against `ALLOWED_MODELS`, not a model ID.
Unsupported `FIREWORKS_EXTRA_BODY` fields are ignored, and mandatory request
fields cannot be overridden by that JSON.

## Runtime contract and safeguards

| Track 1 requirement | Implementation |
|---|---|
| Read `/input/tasks.json` | `app/main.py` loads an array of `{task_id, prompt}` with retry handling. |
| Write `/output/results.json` | A valid `[]` is written immediately; non-empty answers are then written incrementally and atomically before final validation. |
| `linux/amd64`, CPU-only | Build and push targets pin `--platform linux/amd64`; the published v11 image uses llama.cpp CPU inference. |
| 4 GB RAM / 2 vCPU | The pinned 1.5B Q4 GGUF and small JSON router avoid the Torch/DistilBERT runtime and concurrent large-model residency. |
| Startup under 60 seconds | Router scoring and GGUF startup time are reported in stderr as `startup complete in ...`; v11 started in 5.9s in the 80-task rehearsal. |
| Total runtime under 10 minutes | `TOTAL_DEADLINE_S` defaults to 540 seconds, with a final collection/write reserve. |
| Requests under 30 seconds | Local and reactive hosted work share the same 25-second per-task clock; hosted requests have a separate hard ceiling below 30 seconds. |
| Use only allowed hosted models | `app/fireworks.py` chooses exact members of runtime `ALLOWED_MODELS`; unavailable models are removed for the current run. |
| No cached answers or secrets | Answers are generated per prompt. `.dockerignore` excludes `.env`, datasets, tests, and evaluation artifacts. |

The final `done:` stderr line reports answer-path counts, planned escalations,
router state, Fireworks attempts/successes, provider-reported tokens, local
cutoffs, and the category histogram.

Useful runtime controls include `ROUTER_THRESHOLD`, `RESERVE_S`,
`PREFILL_RATIO`, `TASK_OVERHEAD_S`, `TIGHT_ESCALATION_CAP`,
`MIN_LOCAL_TOK_S`, and `ESCALATION_TIMEOUT_S`. Raising the learned threshold
reduces planned escalations but increases the risk of trusting a weak local
answer; it should be changed only after a new calibration run.

## Live demo

The Vercel-ready walkthrough is in [`demo/`](demo/). It runs the same compact
JSON router in the browser, streams a paid prompt-router comparison, returns a
real Fireworks answer in live mode, and displays provider-reported token usage
and session totals. Until `make train-router` emits a valid profile, its run
button remains disabled rather than presenting placeholder routing as learned
behavior.

```bash
cd demo
npm install
npm run dev
```

For Vercel, import this repository and set **Root Directory** to `demo`. See
[`demo/README.md`](demo/README.md) for environment variables, comparison-token
semantics, and the required public rate-limit rule.

## Security note

Never commit either the root `.env` or `demo/.env*` credential files. If a real
Fireworks key was ever pushed in an earlier revision, revoke and rotate it;
replacing the current file does not invalidate a key that remains in Git
history. Never expose the replacement as a `NEXT_PUBLIC_` variable.
