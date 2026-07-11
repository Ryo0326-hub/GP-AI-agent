"""Budget-aware planning: pure functions, zero I/O, fully unit-testable.

The v2 orchestrator plans upfront instead of reacting: tasks the router
predicts the local pipeline will fail are dispatched to Fireworks immediately
and concurrently; everything else is solved locally in ascending estimated
cost, each task admitted only if its estimate still fits the remaining
deadline. There is no per-task time floor — a fast local model simply gets
through more tasks.
"""
import math
import os

from prompts import MAX_TOKENS

# Prefill (prompt ingestion) is much faster than decode on llama.cpp CPU.
PREFILL_RATIO = float(os.environ.get("PREFILL_RATIO", "6"))
# Fixed per-task overhead: sampler setup, verification subprocess, writes.
TASK_OVERHEAD_S = float(os.environ.get("TASK_OVERHEAD_S", "1.5"))
# Headroom for the one verification-driven local retry each solver may take.
RETRY_HEADROOM = {
    "math": 1.7, "debug": 1.9, "codegen": 1.9, "summarization": 1.6,
    "logic": 1.1, "sentiment": 1.1, "ner": 1.1, "factual": 1.1,
}


def estimate_local_cost(prompt, category, tok_s, expected_tokens=None):
    """Seconds to solve a task locally, including likely retries.

    expected_tokens: measured p90 completion tokens for the category (from
    labeling); falls back to the category's hard cap.
    """
    tok_s = max(0.5, float(tok_s))
    prompt_tokens = max(8.0, len(prompt or "") / 4.0)
    measured_latency = 0.0
    measured_completion = False
    raw_completion = expected_tokens
    if isinstance(expected_tokens, dict):
        raw_completion = expected_tokens.get("completion_tokens")
        try:
            measured_latency = float(
                expected_tokens.get("p90_latency_s") or 0.0)
        except (TypeError, ValueError):
            measured_latency = 0.0
    try:
        completion = float(raw_completion)
        measured_completion = math.isfinite(completion) and completion > 0
    except (TypeError, ValueError):
        measured_completion = False
        completion = 0.0
    if not measured_completion:
        completion = float(MAX_TOKENS.get(category, 200))

    # Measured p90 completion tokens already include the solver's retry. Apply
    # retry headroom only when falling back to a single-call hard cap.
    headroom = 1.0 if measured_completion else RETRY_HEADROOM.get(category, 1.2)
    token_estimate = (prompt_tokens / (tok_s * PREFILL_RATIO)
                      + completion * headroom / tok_s
                      + TASK_OVERHEAD_S)
    if math.isfinite(measured_latency) and measured_latency > 0:
        return max(measured_latency, token_estimate)
    return token_estimate


def build_plan(prompts, categories, scores, threshold, policy,
               tok_s=8.0, expected_tokens=None):
    """Split task indices into (escalate_now, local_queue).

    scores: per-task P(escalate) from the router, or None when the router is
    unavailable — then only the always-escalate category policy applies and
    verification remains the safety net for everything else.
    local_queue is sorted by ascending estimated local cost so a tight clock
    banks the cheap wins first.
    """
    always = set((policy or {}).get("always_escalate", ()))
    escalate_now, local = [], []
    for i, (prompt, cat) in enumerate(zip(prompts, categories)):
        score = scores[i] if scores else None
        if cat in always or (score is not None and score >= threshold):
            escalate_now.append(i)
        else:
            local.append(i)
    exp = expected_tokens or {}
    local.sort(key=lambda i: estimate_local_cost(
        prompts[i], categories[i], tok_s, exp.get(categories[i])))
    return escalate_now, local


def fits_deadline(estimate_s, remaining_s, reserve_s):
    """Admission check run before each local task."""
    return estimate_s <= remaining_s - reserve_s


def resolve_threshold(config):
    """ROUTER_THRESHOLD env wins; else the trained config; else 0.5."""
    env = os.environ.get("ROUTER_THRESHOLD")
    def valid(value):
        return (not isinstance(value, bool)
                and isinstance(value, (int, float))
                and math.isfinite(float(value))
                and 0.0 <= float(value) <= 1.01)

    if env:
        try:
            value = float(env)
            if valid(value):
                return value
        except ValueError:
            pass
    if config and valid(config.get("threshold")):
        return float(config["threshold"])
    return 0.5
