"""Track 1 agent orchestrator, v2: learned router + budget-aware plan.

Reads /input/tasks.json, answers every task, writes /output/results.json
incrementally and atomically, and exits 0. All logging goes to stderr.

Flow: score every task with the learned compact router at startup ->
tasks predicted to fail locally are dispatched to Fireworks immediately and
concurrently (category prompt, tight caps) -> everything else is solved locally in
ascending estimated cost, admitted task-by-task against the real clock ->
deterministic verification stays as a cheap second gate, with single-task
escalation (not panic bulk escalation) as the fallback.
"""
import logging
import os
import re
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed, wait

from budget import (build_plan, estimate_local_cost, fits_deadline,
                    resolve_threshold)
from classifier import classify
from fireworks import Fireworks
from prompts import (CODE_REPAIR_SYSTEM, ESCALATION_MAX_TOKENS, MATH_RETRY_SYSTEM,
                     MAX_TOKENS, SYSTEM)
from router import load_config, score_and_free
from utils import (atomic_write_json, extract_labeled_line, extract_last_number,
                   load_tasks, log, log_rss, numbers_equal, setup_logging)
from verify import (check_summary, extract_code, generic_smoke_tests,
                    looks_confident, parse_summary_constraints, run_code_tests,
                    safe_eval, solve_assignment_puzzle, synthesize_tests)

INPUT_PATH = os.environ.get("INPUT_PATH", "/input/tasks.json")
OUTPUT_PATH = os.environ.get("OUTPUT_PATH", "/output/results.json")
MODEL_PATH = os.environ.get("MODEL_PATH", "/models/model.gguf")
TOTAL_DEADLINE_S = float(os.environ.get("TOTAL_DEADLINE_S", "540"))  # 9 min
PER_TASK_CAP_S = float(os.environ.get("PER_TASK_CAP_S", "25"))       # <30s rule
LOCAL_SOLVE_CAP_S = float(os.environ.get("LOCAL_SOLVE_CAP_S", "18"))
MIN_LOCAL_TOK_S = float(os.environ.get("MIN_LOCAL_TOK_S", "4"))       # slow-CPU circuit breaker
RESERVE_S = float(os.environ.get("RESERVE_S", "25"))   # endgame: collect escalations + final write
# The general rules require every request to finish in under 30 seconds. Keep
# a full second of hard headroom even if a development env tries to raise the
# timeout. Queueing and a preceding local attempt are charged to this same
# end-to-end budget by _complete_with_task_deadline below.
MAX_REQUEST_S = 29.0
ESCALATION_TIMEOUT_S = min(
    float(os.environ.get("ESCALATION_TIMEOUT_S", "25")), MAX_REQUEST_S)
TIGHT_ESCALATION_CAP = int(os.environ.get("TIGHT_ESCALATION_CAP", "160"))
ESCALATION_WORKERS = 6
BULK_ABORT_AFTER = 3  # probe wave; only definitive provider failures stop the rest
ESCALATE_UNVERIFIED_LOGIC = os.environ.get("ESCALATE_UNVERIFIED_LOGIC", "1") == "1"
# Accuracy-first mode (judging FAQ: correctness precedes token efficiency).
# When Fireworks is reachable, every task except a deterministically
# brute-forced logic puzzle is answered by the hosted tier; the local pipeline
# remains the full fallback when Fireworks is missing or breaks mid-run.
REMOTE_FIRST = os.environ.get("REMOTE_FIRST", "1") == "1"

FALLBACK_ANSWER = "Unable to fully determine the answer within the time limit."

# Answer paths, for truthful diagnostics.
LOCAL_VERIFIED = "local-verified"
LOCAL_UNVERIFIED = "local-unverified"
ESCALATED = "escalated"
FALLBACK = "fallback"

LAST_RUN_STATS = None  # set by main() for tests/inspection

_WRITE_LOCK = threading.RLock()  # escalation callbacks write concurrently
_FINALIZED = False  # guarded by _WRITE_LOCK; late callbacks become no-ops


def _fmt_num(v):
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    if isinstance(v, float):
        return f"{round(v, 6):g}"
    return str(v)


def _strip_expression_line(text: str) -> str:
    lines = [l for l in (text or "").splitlines()
             if not re.match(r"\s*\**Expression\**\s*[:=]", l, re.IGNORECASE)]
    return "\n".join(lines).strip()


# --------------------------------------------------------------- category solvers
# Each returns (answer_text, verified) where verified=True means we trust the
# answer enough to skip escalation. Each solver includes at most ONE local
# retry on verification failure; after that the orchestrator escalates.

def solve_math(lm, prompt, budget):
    t0 = time.monotonic()
    text = lm.chat(SYSTEM["math"], prompt, MAX_TOKENS["math"],
                   time_budget=max(4.0, budget * 0.55))

    def verdict(t):
        expr = extract_labeled_line(t, "Expression")
        stated = extract_last_number(extract_labeled_line(t, "Answer") or t)
        try:
            val = safe_eval(expr) if expr else None
        except ValueError:
            val = None
        return val, stated, (val is not None and numbers_equal(val, stated))

    val, stated, ok = verdict(text)
    if ok:
        return _strip_expression_line(text), True
    remaining = budget - (time.monotonic() - t0)
    if remaining > 6.0:
        text2 = lm.chat(MATH_RETRY_SYSTEM, prompt, MAX_TOKENS["math"],
                        time_budget=remaining - 1.0)
        val2, stated2, ok2 = verdict(text2)
        if ok2:
            return _strip_expression_line(text2), True
        # Two runs agree on the stated answer -> accept even without expression.
        if stated is not None and stated2 is not None and numbers_equal(stated, stated2):
            return _strip_expression_line(text2), True
        if val2 is not None:
            val, text = val2, text2
    if val is not None:
        # The arithmetic evaluator is exact; prefer its value, but stay unverified.
        return _strip_expression_line(text) + f"\nFinal answer: {_fmt_num(val)}", False
    return _strip_expression_line(text) or "", False


def _describe_failures(report, tests):
    if "error" in report:
        return report["error"]
    parts = []
    for t, r in zip(tests, report.get("results", [])):
        if not r.get("passed"):
            exp = f", expected {t['expected']!r}" if "expected" in t else ""
            got = f", got {r['got']}" if "got" in r else f", raised {r.get('error')}"
            parts.append(f"input {t['args']!r}{exp}{got}")
    return "; ".join(parts) or "tests failed"


def solve_code(lm, prompt, category, budget):
    t0 = time.monotonic()
    text = lm.chat(SYSTEM[category], prompt, MAX_TOKENS[category],
                   time_budget=max(5.0, budget * 0.55))
    code = extract_code(text)
    if not code:
        # Small local models sometimes spend their first generation restating the
        # task and hit the wall-clock cutoff before emitting code. Use the unused
        # half of the task budget for one code-only attempt.
        remaining = budget - (time.monotonic() - t0)
        if remaining > 5.0:
            text2 = lm.chat(CODE_REPAIR_SYSTEM, prompt, MAX_TOKENS[category],
                            time_budget=remaining - 1.0)
            code2 = extract_code(text2)
            if code2:
                text, code = text2, code2
        if not code:
            return text, False
    tests = synthesize_tests(prompt) or generic_smoke_tests(code)
    report = run_code_tests(code, tests)
    passed = report.get("ok") and all(r.get("passed") for r in report.get("results", []))
    if passed:
        return text, True
    remaining = budget - (time.monotonic() - t0)
    if remaining > 8.0:
        repair_user = (f"Task: {prompt}\n\nYour code:\n```python\n{code}\n```\n\n"
                       f"Problem: {_describe_failures(report, tests)}")
        text2 = lm.chat(CODE_REPAIR_SYSTEM, repair_user, MAX_TOKENS[category],
                        time_budget=remaining - 1.0)
        code2 = extract_code(text2)
        if code2:
            report2 = run_code_tests(code2, tests)
            if report2.get("ok") and all(r.get("passed") for r in report2.get("results", [])):
                prose = text.split("```")[0].strip()
                answer = (prose + "\n\n" if prose else "") + f"```python\n{code2}\n```"
                return answer, True
    return text, False


def solve_logic(lm, prompt, budget):
    solved = solve_assignment_puzzle(prompt)
    if solved:
        person, item, assign = solved
        detail = "; ".join(f"{n} has the {it}" for n, it in assign.items())
        return (f"{person} owns the {item}. This is the only assignment "
                f"consistent with all the constraints: {detail}."), True
    text = lm.chat(SYSTEM["logic"], prompt, MAX_TOKENS["logic"],
                   time_budget=max(4.0, budget * 0.9))
    has_answer = bool(extract_labeled_line(text, "Answer"))
    if ESCALATE_UNVERIFIED_LOGIC:
        return text, False
    return text, has_answer and looks_confident(text)


def solve_summary(lm, prompt, budget):
    t0 = time.monotonic()
    text = lm.chat(SYSTEM["summarization"], prompt, MAX_TOKENS["summarization"],
                   time_budget=max(4.0, budget * 0.6))
    constraints = parse_summary_constraints(prompt)
    ok, violations = check_summary(text, constraints)
    if ok and text:
        return text, True
    remaining = budget - (time.monotonic() - t0)
    if remaining > 5.0 and text:
        fix_user = (f"Constraints violated: {'; '.join(violations)}.\n"
                    f"Rewrite this summary so it satisfies the original instruction "
                    f"exactly.\nInstruction: {prompt}\nSummary: {text}")
        text2 = lm.chat("Output only the rewritten summary, meeting every "
                        "constraint exactly.", fix_user,
                        MAX_TOKENS["summarization"], time_budget=remaining - 1.0)
        ok2, _ = check_summary(text2, constraints)
        if ok2 and text2:
            return text2, True
        if text2:
            text = text2
    fixed = _hard_fix_summary(text, constraints)
    ok3, _ = check_summary(fixed, constraints)
    return fixed, ok3 and bool(fixed)


def _hard_fix_summary(text, constraints):
    """Code-level constraint enforcement as a last resort."""
    t = (text or "").strip()
    if constraints.get("sentences_exact") == 1 and t:
        # Merge sentences: internal terminators become semicolons.
        t = t.rstrip(".!?")
        t = re.sub(r"[.!?]+\s+", "; ", t) + "."
    if "words_max" in constraints:
        words = t.split()
        if len(words) > constraints["words_max"]:
            t = " ".join(words[:constraints["words_max"]]).rstrip(",;:") + "."
    return t


_SENTIMENT_LABEL = re.compile(r"\b(positive|negative|neutral|mixed)\b", re.IGNORECASE)
_NER_ENTRY = re.compile(
    r"^.+\s[-–—]\s*(person|organi[sz]ation|location|date)\s*$", re.IGNORECASE
)


def solve_simple(lm, prompt, category, budget):
    text = lm.chat(SYSTEM[category], prompt, MAX_TOKENS[category],
                   time_budget=max(4.0, budget * 0.9))
    if category == "sentiment":
        # If the model's explanation explicitly identifies both polarities, its
        # leading label is occasionally inconsistent. Normalize the label from
        # its own evidence rather than accepting a self-contradictory answer.
        lower = text.lower()
        if "both positive" in lower and "negative" in lower:
            text = _SENTIMENT_LABEL.sub("Mixed", text, count=1)
        return text, bool(_SENTIMENT_LABEL.search(text)) and looks_confident(text, 10)
    if category == "ner":
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return text, bool(lines) and all(_NER_ENTRY.fullmatch(line) for line in lines)
    return text, looks_confident(text)  # factual


def solve_task(lm, prompt, category, budget):
    if category == "math":
        return solve_math(lm, prompt, budget)
    if category in ("debug", "codegen"):
        return solve_code(lm, prompt, category, budget)
    if category == "logic":
        return solve_logic(lm, prompt, budget)
    if category == "summarization":
        return solve_summary(lm, prompt, budget)
    return solve_simple(lm, prompt, category, budget)


# ------------------------------------------------------------------- orchestration

def _write_partial_locked(tasks, answers):
    """Write the current snapshot while the caller holds ``_WRITE_LOCK``."""
    results = [{"task_id": str(t.get("task_id", i)), "answer": answers[i]}
               for i, t in enumerate(tasks) if answers.get(i)]
    atomic_write_json(OUTPUT_PATH, results)


def write_partial(tasks, answers):
    """Write a partial snapshot unless final output has been committed."""
    with _WRITE_LOCK:
        if _FINALIZED:
            return False
        _write_partial_locked(tasks, answers)
        return True


def record(i, answer, path, tasks, answers, paths, overwrite=True):
    """Thread-safe result recording + incremental write."""
    with _WRITE_LOCK:
        if _FINALIZED:
            return False
        if not overwrite and (answers.get(i) or "").strip():
            return False
        answers[i] = answer
        paths[i] = path
        # Keep the state mutation and snapshot write in one critical section.
        # Otherwise finalization can land between them and a late callback can
        # replace the complete file with an incomplete partial snapshot.
        _write_partial_locked(tasks, answers)
        return True


def _remote_verified(category, prompt, text):
    """Deterministic second gate for hosted answers, mirroring the local one.

    Categories without an exact check return True: a failed retry there would
    only replay the same temperature-0 request.
    """
    if not (text or "").strip():
        return False
    if category == "math":
        expr = extract_labeled_line(text, "Expression")
        stated = extract_last_number(extract_labeled_line(text, "Answer") or text)
        try:
            val = safe_eval(expr) if expr else None
        except ValueError:
            val = None
        return val is not None and stated is not None and numbers_equal(val, stated)
    if category in ("codegen", "debug"):
        code = extract_code(text)
        if not code:
            return False
        tests = synthesize_tests(prompt) or generic_smoke_tests(code)
        if not tests:
            return True
        report = run_code_tests(code, tests)
        return bool(report.get("ok")) and all(
            r.get("passed") for r in report.get("results", []))
    if category == "summarization":
        return check_summary(text, parse_summary_constraints(prompt))[0]
    if category == "sentiment":
        return bool(_SENTIMENT_LABEL.search(text))
    return True


def _postprocess_remote(category, text):
    if category == "math":
        return _strip_expression_line(text)
    return (text or "").strip()


_AGREE_STOP = frozenset(
    "the a an is are was were of in on at to and or it its by with for as "
    "that this near located also which".split())


def _answers_agree(a, b):
    """Loose semantic agreement: matching final numbers plus high overlap of
    content words. Verbose-vs-terse phrasings of the same fact still agree;
    different named facts (e.g. two different lakes) do not."""
    na, nb = extract_last_number(a), extract_last_number(b)
    if na is not None and nb is not None and not numbers_equal(na, nb):
        return False
    wa = set(re.findall(r"[a-z0-9']+", (a or "").lower())) - _AGREE_STOP
    wb = set(re.findall(r"[a-z0-9']+", (b or "").lower())) - _AGREE_STOP
    if not wa or not wb:
        return False
    return len(wa & wb) / min(len(wa), len(wb)) >= 0.75


_ARBITER_SYSTEM = ("You are verifying factual answers. Reply with exactly one "
                   "letter: A or B.")


def _factual_consensus(fw, prompt, cap, system, time_left):
    """Cross-check a factual answer between two allowed models.

    Factual is the only category without a deterministic verifier, and it is
    where the hosted tier was observed to hallucinate. Disagreement is settled
    by a third allowed model voting A/B; ties fall to the second (usually
    larger) model. Every call stays inside the caller's remaining time.
    """
    t_end = time.monotonic() + time_left
    a = fw.complete(prompt, cap, timeout=max(1.0, t_end - time.monotonic()),
                    system=system)
    second = fw.secondary_model()
    if not second or t_end - time.monotonic() < 4.0:
        return a
    b = fw.complete(prompt, cap, timeout=max(1.0, t_end - time.monotonic()),
                    system=system, model_override=second)
    if not (b or "").strip():
        return a
    if not (a or "").strip() or _answers_agree(a, b):
        return a if (a or "").strip() else b
    arbiter = fw.pick_distinct({fw.model, second}, ())
    if arbiter and t_end - time.monotonic() >= 3.0:
        verdict = fw.complete(
            f"Question: {prompt}\n\nAnswer A: {a}\n\nAnswer B: {b}\n\n"
            "Which answer is more factually accurate and complete?",
            10, timeout=max(1.0, t_end - time.monotonic()),
            system=_ARBITER_SYSTEM, model_override=arbiter)
        m = re.search(r"\b([AB])\b", verdict or "")
        if m:
            log.info("factual consensus: arbiter %s chose %s", arbiter, m.group(1))
            return a if m.group(1) == "A" else b
    # No usable arbiter: prefer the larger knowledge model's answer.
    return b


def _complete_with_task_deadline(fw, prompt, cap, task_started_at, deadline,
                                 system=None, category=None):
    """Call Fireworks within the task's real end-to-end time budget.

    This function runs inside the executor worker, so time spent waiting in
    the executor queue is included.  For reactive escalations,
    ``task_started_at`` is the start of the local attempt, preventing an 18s
    local solve followed by a fresh 25s remote allowance.

    When ``category`` is given, the answer must pass the same deterministic
    checks applied to local answers; a failure buys one retry on a different
    allowed model inside the unchanged per-task clock.
    """
    now = time.monotonic()
    per_task_budget = min(PER_TASK_CAP_S, MAX_REQUEST_S)
    remaining = min(
        ESCALATION_TIMEOUT_S,
        task_started_at + per_task_budget - now,
        deadline - now - 2.0,
    )
    if remaining < 1.0:
        log.warning("skipping escalation: task deadline exhausted in queue")
        return ""
    if category == "factual" and getattr(fw, "pick_distinct", None):
        return _postprocess_remote(
            category, _factual_consensus(fw, prompt, cap, system, remaining))
    text = fw.complete(prompt, cap, timeout=remaining, system=system)
    if category is None or _remote_verified(category, prompt, text):
        return _postprocess_remote(category, text)
    retry_model = getattr(fw, "secondary_model", lambda: None)()
    now = time.monotonic()
    remaining = min(
        ESCALATION_TIMEOUT_S,
        task_started_at + per_task_budget - now,
        deadline - now - 2.0,
    )
    if retry_model and remaining >= 4.0:
        log.info("hosted answer failed %s verification; retrying on %s",
                 category, retry_model)
        text2 = fw.complete(prompt, cap, timeout=remaining, system=system,
                            model_override=retry_model)
        if _remote_verified(category, prompt, text2):
            return _postprocess_remote(category, text2)
        if not (text or "").strip():
            text = text2
    return _postprocess_remote(category, text)


def submit_escalations(ex, fw, tasks, categories, indices, answers, paths,
                       deadline, cap_override=None, task_starts=None):
    """Dispatch tasks to Fireworks concurrently; results land via callbacks
    the moment each completes, so partial output is always current.

    ``task_starts`` optionally maps an index to the beginning of its local
    attempt. Missing entries start their end-to-end clock when queued.
    """
    futures = {}
    for i in indices:
        prompt = str(tasks[i].get("prompt", ""))
        cap = cap_override or ESCALATION_MAX_TOKENS.get(categories[i], 300)
        system = SYSTEM.get(categories[i])
        queued_at = time.monotonic()
        task_started_at = (task_starts or {}).get(i, queued_at)
        fut = ex.submit(_complete_with_task_deadline, fw, prompt, cap,
                        task_started_at, deadline, system, categories[i])

        def _cb(f, i=i):
            try:
                ans = f.result() or ""
            except Exception as e:  # noqa: BLE001
                log.warning("escalation for task %d failed: %s", i, e)
                ans = ""
            if ans.strip():
                record(i, ans, ESCALATED, tasks, answers, paths, overwrite=False)

        fut.add_done_callback(_cb)
        futures[fut] = i
    return futures


def _escalate_batch(ex, fw, tasks, indices, answers, paths, deadline):
    """Run one concurrent escalation wave; returns the number of successes."""
    futures = {}
    for i in indices:
        prompt = str(tasks[i].get("prompt", ""))
        category = classify(prompt)
        cap = ESCALATION_MAX_TOKENS.get(category, 300)
        system = SYSTEM.get(category)
        task_started_at = time.monotonic()
        futures[ex.submit(_complete_with_task_deadline, fw, prompt, cap,
                          task_started_at, deadline, system, category)] = i
    successes = 0
    timeout = deadline - time.monotonic() - 2.0
    if timeout <= 0:
        for future in futures:
            future.cancel()
        return 0
    try:
        for fut in as_completed(futures, timeout=timeout):
            i = futures[fut]
            try:
                answer = fut.result() or ""
            except Exception as e:  # noqa: BLE001
                log.warning("bulk escalation task %d failed: %s", i, e)
                answer = ""
            if answer:
                record(i, answer, ESCALATED, tasks, answers, paths, overwrite=False)
                successes += 1
    except TimeoutError:
        log.warning("bulk escalation hit the deadline")
        for f in futures:
            f.cancel()
    return successes


def bulk_escalate(fw, tasks, pending, answers, paths, deadline, executor=None):
    """Escalate all pending tasks concurrently (used only when the local model
    is unusable). Probes the first BULK_ABORT_AFTER tasks first, but abandons
    the rest only after a definitive provider/configuration failure. Transient
    probe failures continue so a brief 429/5xx wave cannot create mass
    fallbacks. Returns the indices still unanswered."""
    log.info("bulk escalation of %d remaining tasks", len(pending))
    probes, rest = pending[:BULK_ABORT_AFTER], pending[BULK_ABORT_AFTER:]

    def run(ex):
        probe_successes = _escalate_batch(ex, fw, tasks, probes, answers, paths, deadline)
        definitive_failure = (
            not getattr(fw, "available", False)
            or getattr(fw, "definitive_unavailable", False)
        )
        if probe_successes == 0 and definitive_failure:
            log.warning("first %d escalations all failed; abandoning bulk "
                        "after a definitive provider failure", len(probes))
        elif rest:
            if probe_successes == 0:
                log.warning("initial escalation probes failed transiently; "
                            "continuing instead of creating mass fallbacks")
            _escalate_batch(ex, fw, tasks, rest, answers, paths, deadline)
        return [i for i in pending if paths.get(i) != ESCALATED]

    if executor is not None:
        return run(executor)
    with ThreadPoolExecutor(max_workers=ESCALATION_WORKERS) as owned_executor:
        return run(owned_executor)


def main() -> int:
    global LAST_RUN_STATS, _FINALIZED
    with _WRITE_LOCK:
        _FINALIZED = False
    setup_logging()
    start = time.monotonic()
    deadline = start + TOTAL_DEADLINE_S
    log.info("agent v2 starting (deadline %.0fs, local cap %.0fs, reserve %.0fs)",
             TOTAL_DEADLINE_S, LOCAL_SOLVE_CAP_S, RESERVE_S)

    try:
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    except OSError:
        pass

    tasks = load_tasks(INPUT_PATH)
    n = len(tasks)
    prompts = [str(t.get("prompt", "")) for t in tasks]
    categories = [classify(p) for p in prompts]
    cat_counts = Counter(categories)
    log.info("loaded %d tasks: %s", n, dict(cat_counts))
    answers = {}
    paths = {}  # index -> LOCAL_VERIFIED | LOCAL_UNVERIFIED | ESCALATED | FALLBACK
    atomic_write_json(OUTPUT_PATH, [])  # valid JSON exists from second zero
    log_rss("startup")

    # ---- router scoring (zero tokens; weights freed before llama loads) ----
    config = load_config()
    threshold = resolve_threshold(config)
    policy = (config or {}).get("category_policy") or {}
    trust_local = set(policy.get("trust_local") or ())
    expected_tokens = (config or {}).get("expected_completion_tokens") or {}
    scores = score_and_free(prompts) if n else []
    log_rss("router scored")

    fw = Fireworks()
    log.info("fireworks available=%s model=%s allowed=%d",
             fw.available, fw.model, len(fw.allowed))

    # Zero-token exact wins: brute-forced assignment puzzles need no model at
    # all, and the enumeration proves the answer unique before it is trusted.
    pre_solved = set()
    for i, (p, cat) in enumerate(zip(prompts, categories)):
        if cat != "logic":
            continue
        solved = solve_assignment_puzzle(p)
        if solved:
            person, item, assign = solved
            detail = "; ".join(f"{n} has the {it}" for n, it in assign.items())
            record(i, (f"{person} owns the {item}. This is the only assignment "
                       f"consistent with all the constraints: {detail}."),
                   LOCAL_VERIFIED, tasks, answers, paths)
            pre_solved.add(i)
    if pre_solved:
        log.info("%d logic task(s) solved exactly by enumeration", len(pre_solved))

    escalate_now, local_queue = build_plan(prompts, categories, scores,
                                           threshold, policy,
                                           expected_tokens=expected_tokens)
    if REMOTE_FIRST and fw.available:
        escalate_now = [i for i in range(n) if i not in pre_solved]
        local_queue = []
        log.info("remote-first mode: hosted tier answers all %d unsolved tasks",
                 len(escalate_now))
    if not fw.available and escalate_now:
        log.warning("no Fireworks path; the %d planned escalations go local",
                    len(escalate_now))
        escalate_now, local_queue = build_plan(prompts, categories, None, 2.0,
                                               {}, expected_tokens=expected_tokens)
    escalate_now = [i for i in escalate_now if i not in pre_solved]
    local_queue = [i for i in local_queue if i not in pre_solved]
    log.info("plan: %d immediate escalations, %d local, %d pre-solved "
             "(router=%s threshold=%.2f policy=%s)",
             len(escalate_now), len(local_queue), len(pre_solved),
             "on" if scores else "off", threshold, policy or "{}")

    ex = ThreadPoolExecutor(max_workers=ESCALATION_WORKERS)
    futures = {}
    if escalate_now:
        futures.update(submit_escalations(ex, fw, tasks, categories,
                                          escalate_now, answers, paths, deadline))

    # ---- local model (loads while the planned escalations are in flight) ----
    lm = None
    emergency_lm = None
    try:
        from local_model import LocalLM
        lm = LocalLM(MODEL_PATH)
        emergency_lm = lm
        speed = lm.benchmark()
        log_rss("llama loaded")
        log.info("startup complete in %.1fs (constraint: <60s)",
                 time.monotonic() - start)
        if fw.available and speed < MIN_LOCAL_TOK_S:
            log.warning("local model too slow (%.1f < %.1f tok/s); using Fireworks "
                        "to protect latency and accuracy", speed, MIN_LOCAL_TOK_S)
            lm = None
    except Exception as e:  # noqa: BLE001
        log.error("local model unavailable, will escalate everything: %s", e)
        lm = None
        emergency_lm = None

    backup = {}  # local answers kept as fallback for in-flight escalations

    if lm is None:
        if fw.available and local_queue:
            bulk_escalate(fw, tasks, local_queue, answers, paths, deadline,
                          executor=ex)
    else:
        # Re-rank with the measured speed, then admit against the real clock.
        local_queue.sort(key=lambda i: estimate_local_cost(
            prompts[i], categories[i], lm.tok_per_sec,
            expected_tokens.get(categories[i])))
        reserve = RESERVE_S if fw.available else 5.0
        for pos, i in enumerate(local_queue):
            remaining = deadline - time.monotonic()
            est = estimate_local_cost(prompts[i], categories[i], lm.tok_per_sec,
                                      expected_tokens.get(categories[i]))
            if not fits_deadline(est, remaining, reserve):
                if fw.available:
                    log.warning("task %d no longer fits locally (est %.0fs, "
                                "%.0fs left); escalating with tight cap", i, est,
                                remaining)
                    futures.update(submit_escalations(
                        ex, fw, tasks, categories, [i], answers, paths,
                        deadline, cap_override=TIGHT_ESCALATION_CAP))
                    continue
                # No escalation path: degrade budgets, never skip.
                todo_left = len(local_queue) - pos
                budget = max(3.0, min(PER_TASK_CAP_S,
                                      (remaining - 5.0) / max(1, todo_left)))
            else:
                budget = min(PER_TASK_CAP_S, LOCAL_SOLVE_CAP_S,
                             max(4.0, remaining - reserve))
            t0 = time.monotonic()
            try:
                answer, verified = solve_task(lm, prompts[i], categories[i], budget)
            except Exception as e:  # noqa: BLE001
                log.warning("task %d local solve error: %s", i, e)
                answer, verified = "", False
            if verified:
                record(i, answer, LOCAL_VERIFIED, tasks, answers, paths)
            elif (answer or "").strip() and categories[i] in trust_local:
                record(i, answer, LOCAL_UNVERIFIED, tasks, answers, paths)
            elif fw.available:
                if (answer or "").strip():
                    backup[i] = answer
                futures.update(submit_escalations(ex, fw, tasks, categories,
                                                  [i], answers, paths, deadline,
                                                  task_starts={i: t0}))
            elif (answer or "").strip():
                record(i, answer, LOCAL_UNVERIFIED, tasks, answers, paths)
            else:
                record(i, FALLBACK_ANSWER, FALLBACK, tasks, answers, paths)
            log.info("task %d/%d [%s] path=%s in %.1fs (budget %.0fs est %.0fs)",
                     i + 1, n, categories[i], paths.get(i, "escalating"),
                     time.monotonic() - t0, budget, est)

    # ---- endgame: collect in-flight escalations, then fill any gaps ----
    if futures:
        wait_s = max(0.0, deadline - time.monotonic() - 3.0)
        done, not_done = wait(list(futures), timeout=wait_s)
        for f in not_done:
            f.cancel()
        # Callbacks normally record results; drain directly as a belt-and-braces
        # against a callback that hasn't run yet.
        for f in done:
            i = futures[f]
            try:
                ans = f.result() or ""
            except Exception:  # noqa: BLE001
                ans = ""
            if ans.strip():
                record(i, ans, ESCALATED, tasks, answers, paths, overwrite=False)
    ex.shutdown(wait=False, cancel_futures=True)

    with _WRITE_LOCK:
        unanswered = [i for i in range(n) if not (answers.get(i) or "").strip()]
    for i in unanswered:
        if backup.get(i):
            record(i, backup[i], LOCAL_UNVERIFIED, tasks, answers, paths,
                   overwrite=False)
        elif emergency_lm is not None and deadline - time.monotonic() > 10.0:
            budget = max(3.0, min(LOCAL_SOLVE_CAP_S,
                                  deadline - time.monotonic() - 5.0))
            try:
                answer, verified = solve_task(
                    emergency_lm, prompts[i], categories[i], budget)
            except Exception:  # noqa: BLE001
                answer, verified = "", False
            if (answer or "").strip():
                record(i, answer,
                       LOCAL_VERIFIED if verified else LOCAL_UNVERIFIED,
                       tasks, answers, paths, overwrite=False)

    # Final validation: every task_id present, every answer a non-empty string.
    with _WRITE_LOCK:
        results = []
        for i, t in enumerate(tasks):
            a = answers.get(i)
            if not isinstance(a, str) or not a.strip():
                a = FALLBACK_ANSWER
                answers[i] = a
                paths[i] = FALLBACK
            paths.setdefault(i, FALLBACK)
            results.append({"task_id": str(t.get("task_id", i)), "answer": a})
        # Flip the guard before the final write while holding the lock. Any
        # running provider callback that completes later will observe this and
        # cannot replace the complete file with a partial snapshot.
        _FINALIZED = True
        atomic_write_json(OUTPUT_PATH, results)

    path_counts = Counter(paths[i] for i in range(n))
    cutoffs = emergency_lm.cutoff_count if emergency_lm is not None else 0
    LAST_RUN_STATS = {"paths": dict(path_counts), "categories": dict(cat_counts),
                      "fw_attempted": fw.calls_attempted,
                      "fw_succeeded": fw.calls_succeeded,
                      "fw_tokens": fw.total_tokens, "cutoffs": cutoffs,
                      "fw_usage_derived": getattr(
                          fw, "derived_usage_calls", 0),
                      "fw_usage_unknown": getattr(
                          fw, "unknown_usage_calls", 0),
                      "router_scored": bool(scores),
                      "planned_escalations": len(escalate_now),
                      "threshold": threshold}
    log.info("done: answered=%d in %.1fs | local-verified=%d local-unverified=%d "
             "escalated=%d fallback=%d | planned-escalations=%d router=%s | "
             "fireworks attempted=%d succeeded=%d known-tokens=%d "
             "usage-derived=%d usage-unknown=%d | cutoffs=%d | categories=%s",
             n, time.monotonic() - start,
             path_counts.get(LOCAL_VERIFIED, 0), path_counts.get(LOCAL_UNVERIFIED, 0),
             path_counts.get(ESCALATED, 0), path_counts.get(FALLBACK, 0),
             len(escalate_now), "on" if scores else "off",
             fw.calls_attempted, fw.calls_succeeded, fw.total_tokens,
             getattr(fw, "derived_usage_calls", 0),
             getattr(fw, "unknown_usage_calls", 0), cutoffs, dict(cat_counts))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:  # noqa: BLE001 — last-ditch: still emit valid JSON
        logging.getLogger("agent").exception("fatal: %s", e)
        try:
            tasks = load_tasks(INPUT_PATH, retries=1)
            atomic_write_json(OUTPUT_PATH, [
                {"task_id": str(t.get("task_id", i)), "answer": FALLBACK_ANSWER}
                for i, t in enumerate(tasks)])
            sys.exit(0)
        except Exception:  # noqa: BLE001
            atomic_write_json(OUTPUT_PATH, [])
            sys.exit(1)
