"""Track 1 agent orchestrator.

Reads /input/tasks.json, answers every task (local-first, verified, rare
Fireworks escalation), writes /output/results.json incrementally and
atomically, and exits 0. All logging goes to stderr.
"""
import logging
import os
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from classifier import classify
from fireworks import Fireworks
from prompts import (CODE_REPAIR_SYSTEM, ESCALATION_MAX_TOKENS, MATH_RETRY_SYSTEM,
                     MAX_TOKENS, SYSTEM)
from utils import (atomic_write_json, extract_labeled_line, extract_last_number,
                   load_tasks, log, numbers_equal, setup_logging)
from verify import (check_summary, extract_code, generic_smoke_tests,
                    looks_confident, parse_summary_constraints, run_code_tests,
                    safe_eval, solve_assignment_puzzle, synthesize_tests)

INPUT_PATH = os.environ.get("INPUT_PATH", "/input/tasks.json")
OUTPUT_PATH = os.environ.get("OUTPUT_PATH", "/output/results.json")
MODEL_PATH = os.environ.get("MODEL_PATH", "/models/model.gguf")
TOTAL_DEADLINE_S = float(os.environ.get("TOTAL_DEADLINE_S", "540"))  # 9 min
PER_TASK_CAP_S = float(os.environ.get("PER_TASK_CAP_S", "25"))       # <30s rule
PER_TASK_MIN_S = float(os.environ.get("PER_TASK_MIN_S", "20"))       # budget floor
LOCAL_SOLVE_CAP_S = float(os.environ.get("LOCAL_SOLVE_CAP_S", "18")) # reserve escalation time
MIN_LOCAL_TOK_S = float(os.environ.get("MIN_LOCAL_TOK_S", "4"))       # slow-CPU circuit breaker
BULK_ABORT_AFTER = 3  # consecutive initial bulk-escalation failures before giving up
ESCALATE_UNVERIFIED_LOGIC = os.environ.get("ESCALATE_UNVERIFIED_LOGIC", "1") == "1"

FALLBACK_ANSWER = "Unable to fully determine the answer within the time limit."

# Answer paths, for truthful diagnostics.
LOCAL_VERIFIED = "local-verified"
LOCAL_UNVERIFIED = "local-unverified"
ESCALATED = "escalated"
FALLBACK = "fallback"

LAST_RUN_STATS = None  # set by main() for tests/inspection


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
# answer enough to skip escalation.

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

def plan_budget(remaining: float, todo: int, fw_available: bool):
    """Per-task time policy.

    Returns ("bulk", None) when there isn't enough time to give every remaining
    task at least PER_TASK_MIN_S locally and Fireworks can take over; otherwise
    ("local", budget) with budget clamped to [PER_TASK_MIN_S, PER_TASK_CAP_S]
    (degraded below the floor only when there is no escalation path at all —
    a squeezed local answer beats no answer).
    """
    todo = max(1, todo)
    if remaining < todo * PER_TASK_MIN_S + 15.0:
        if fw_available:
            return "bulk", None
        return "local", max(4.0, remaining / todo)
    return "local", min(PER_TASK_CAP_S, max(PER_TASK_MIN_S, remaining / todo))


def write_partial(tasks, answers):
    results = [{"task_id": str(t.get("task_id", i)), "answer": answers[i]}
               for i, t in enumerate(tasks) if answers.get(i)]
    atomic_write_json(OUTPUT_PATH, results)


def _escalate_batch(ex, fw, tasks, indices, answers, paths, deadline):
    """Run one concurrent escalation wave; returns the number of successes."""
    futures = {}
    for i in indices:
        prompt = str(tasks[i].get("prompt", ""))
        cap = ESCALATION_MAX_TOKENS.get(classify(prompt), 300)
        futures[ex.submit(fw.complete, prompt, cap)] = i
    successes = 0
    try:
        for fut in as_completed(futures, timeout=max(10.0, deadline - time.monotonic())):
            i = futures[fut]
            try:
                answer = fut.result() or ""
            except Exception as e:  # noqa: BLE001
                log.warning("bulk escalation task %d failed: %s", i, e)
                answer = ""
            if answer:
                answers[i] = answer
                paths[i] = ESCALATED
                successes += 1
                write_partial(tasks, answers)
    except TimeoutError:
        log.warning("bulk escalation hit the deadline")
        for f in futures:
            f.cancel()
    return successes


def bulk_escalate(fw, tasks, pending, answers, paths, deadline):
    """Escalate all pending tasks concurrently; a few hundred tokens beat a timeout.

    Probes the first BULK_ABORT_AFTER tasks first: if every probe fails, the
    endpoint is dead and the remaining tasks are never attempted (no burning
    2 HTTP calls per task on a dead route). Returns the indices still
    unanswered so the caller can fall back to local answers.
    """
    log.info("bulk escalation of %d remaining tasks", len(pending))
    probes, rest = pending[:BULK_ABORT_AFTER], pending[BULK_ABORT_AFTER:]
    with ThreadPoolExecutor(max_workers=6) as ex:
        probe_successes = _escalate_batch(ex, fw, tasks, probes, answers, paths, deadline)
        if probe_successes == 0:
            log.warning("first %d escalations all failed; abandoning bulk "
                        "escalation, falling back to local answers", len(probes))
        elif rest:
            _escalate_batch(ex, fw, tasks, rest, answers, paths, deadline)
    return [i for i in pending if paths.get(i) != ESCALATED]


def main() -> int:
    global LAST_RUN_STATS
    setup_logging()
    start = time.monotonic()
    deadline = start + TOTAL_DEADLINE_S
    log.info("agent starting (deadline %.0fs, per-task floor %.0fs cap %.0fs)",
             TOTAL_DEADLINE_S, PER_TASK_MIN_S, PER_TASK_CAP_S)

    try:
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    except OSError:
        pass

    tasks = load_tasks(INPUT_PATH)
    n = len(tasks)
    categories = Counter(classify(str(t.get("prompt", ""))) for t in tasks)
    log.info("loaded %d tasks: %s", n, dict(categories))
    answers = {}
    paths = {}  # index -> LOCAL_VERIFIED | LOCAL_UNVERIFIED | ESCALATED | FALLBACK
    atomic_write_json(OUTPUT_PATH, [])  # valid JSON exists from second zero

    fw = Fireworks()
    log.info("fireworks available=%s model=%s allowed=%d",
             fw.available, fw.model, len(fw.allowed))

    lm = None
    try:
        from local_model import LocalLM
        lm = LocalLM(MODEL_PATH)
        speed = lm.benchmark()
        if fw.available and speed < MIN_LOCAL_TOK_S:
            log.warning("local model too slow (%.1f < %.1f tok/s); using Fireworks "
                        "to protect latency and accuracy", speed, MIN_LOCAL_TOK_S)
            lm = None
    except Exception as e:  # noqa: BLE001
        log.error("local model unavailable, will escalate everything: %s", e)

    pending = list(range(n))

    if lm is not None:
        for i in list(pending):
            remaining = deadline - time.monotonic()
            mode, budget = plan_budget(remaining, len(pending), fw.available)
            if mode == "bulk":
                log.warning("behind pace (%.0fs for %d tasks, floor %.0fs) -> "
                            "bulk escalation", remaining, len(pending), PER_TASK_MIN_S)
                break
            prompt = str(tasks[i].get("prompt", ""))
            category = classify(prompt)
            t0 = time.monotonic()
            local_budget = min(budget, LOCAL_SOLVE_CAP_S)
            try:
                answer, verified = solve_task(lm, prompt, category, local_budget)
            except Exception as e:  # noqa: BLE001
                log.warning("task %d local solve error: %s", i, e)
                answer, verified = "", False
            if verified:
                paths[i] = LOCAL_VERIFIED
            else:
                esc = ""
                # PER_TASK_CAP_S is end-to-end: local attempts and escalation
                # share one budget rather than each receiving a fresh timeout.
                escalation_budget = budget - (time.monotonic() - t0) - 0.5
                if fw.available and escalation_budget >= 1.0:
                    esc = fw.complete(
                        prompt,
                        ESCALATION_MAX_TOKENS.get(category, 300),
                        timeout=escalation_budget,
                    )
                if esc:
                    answer = esc
                    paths[i] = ESCALATED
                elif (answer or "").strip():
                    paths[i] = LOCAL_UNVERIFIED
                else:
                    answer = FALLBACK_ANSWER
                    paths[i] = FALLBACK
            answers[i] = answer
            pending.remove(i)
            write_partial(tasks, answers)
            log.info("task %d/%d [%s] path=%s in %.1fs (local/total budget %.0f/%.0fs)",
                     i + 1, n, category, paths[i], time.monotonic() - t0,
                     local_budget, budget)

    if pending:
        still = list(pending)
        if fw.available:
            still = bulk_escalate(fw, tasks, pending, answers, paths, deadline)
        if still and lm is not None:
            log.info("answering %d remaining tasks locally (degraded budgets)", len(still))
            for i in still:
                remaining = deadline - time.monotonic()
                budget = max(3.0, min(PER_TASK_CAP_S, remaining / max(1, len(still))))
                prompt = str(tasks[i].get("prompt", ""))
                category = classify(prompt)
                try:
                    answer, verified = solve_task(lm, prompt, category, budget)
                except Exception:  # noqa: BLE001
                    answer, verified = "", False
                if (answer or "").strip():
                    paths[i] = LOCAL_VERIFIED if verified else LOCAL_UNVERIFIED
                    answers[i] = answer
                write_partial(tasks, answers)

    # Final validation: every task_id present, every answer a non-empty string.
    results = []
    for i, t in enumerate(tasks):
        a = answers.get(i)
        if not isinstance(a, str) or not a.strip():
            a = FALLBACK_ANSWER
            paths[i] = FALLBACK
        paths.setdefault(i, FALLBACK)
        results.append({"task_id": str(t.get("task_id", i)), "answer": a})
    atomic_write_json(OUTPUT_PATH, results)

    path_counts = Counter(paths[i] for i in range(n))
    cutoffs = lm.cutoff_count if lm is not None else 0
    LAST_RUN_STATS = {"paths": dict(path_counts), "categories": dict(categories),
                      "fw_attempted": fw.calls_attempted,
                      "fw_succeeded": fw.calls_succeeded,
                      "fw_tokens": fw.total_tokens, "cutoffs": cutoffs}
    log.info("done: answered=%d in %.1fs | local-verified=%d local-unverified=%d "
             "escalated=%d fallback=%d | fireworks attempted=%d succeeded=%d "
             "tokens=%d | cutoffs=%d | categories=%s",
             n, time.monotonic() - start,
             path_counts.get(LOCAL_VERIFIED, 0), path_counts.get(LOCAL_UNVERIFIED, 0),
             path_counts.get(ESCALATED, 0), path_counts.get(FALLBACK, 0),
             fw.calls_attempted, fw.calls_succeeded, fw.total_tokens,
             cutoffs, dict(categories))
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
