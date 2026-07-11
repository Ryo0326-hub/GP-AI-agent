#!/usr/bin/env python3
"""Dev-only judge: scores test_output/results.json against test_input/tasks.json.

Deterministic checks where eval/expected.json has ground truth (numbers, labels,
code tests, containment); otherwise an LLM judge via your own Fireworks key
from .env. Prints per-category accuracy, overall %, and the projected result
against the 80% gate on a 19-task sample.
"""
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "app"))
from utils import extract_last_number, numbers_equal  # noqa: E402
from verify import extract_code, run_code_tests  # noqa: E402

def _pick_judge_model():
    """JUDGE_MODEL env wins; else the strongest-looking model the account can
    actually use (from ALLOWED_MODELS); else a common default."""
    explicit = os.environ.get("JUDGE_MODEL")
    if explicit:
        return explicit
    allowed = [m.strip() for m in (os.environ.get("ALLOWED_MODELS") or "").split(",")
               if m.strip()]
    for hint in ("pro", "70b", "k2", "large", "405b"):
        for m in allowed:
            if hint in m.lower():
                return m
    return allowed[-1] if allowed else "accounts/fireworks/models/llama-v3p3-70b-instruct"


def load_env():
    path = os.path.join(ROOT, ".env")
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def llm_judge(prompt, answer):
    import requests
    base = (os.environ.get("FIREWORKS_BASE_URL") or "https://api.fireworks.ai/inference/v1").rstrip("/")
    key = os.environ.get("FIREWORKS_API_KEY")
    if not key:
        return None, "no FIREWORKS_API_KEY in .env; skipped"
    judge_prompt = (
        "You are grading an AI assistant's answer for correctness and intent.\n"
        f"TASK: {prompt}\n\nANSWER: {answer}\n\n"
        "Is the answer correct and does it fulfill the task's intent (including any "
        "format/length constraints)? Reply with only a JSON object: "
        '{"correct": true or false, "reason": "one short sentence"}')
    try:
        r = requests.post(base + "/chat/completions", timeout=45,
                          headers={"Authorization": f"Bearer {key}"},
                          json={"model": _pick_judge_model(), "temperature": 0, "max_tokens": 500,
                                "messages": [
                                    {"role": "system", "content": "Return only valid JSON."},
                                    {"role": "user", "content": judge_prompt},
                                ]})
        r.raise_for_status()
        message = r.json()["choices"][0]["message"]
        text = message.get("content") or message.get("reasoning_content") or ""
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return None, "judge returned no JSON verdict"
        verdict = json.loads(m.group(0))
        return bool(verdict.get("correct")), verdict.get("reason", "")
    except Exception as e:  # noqa: BLE001
        return None, f"judge error: {e}"


def deterministic_check(kind, expected, answer):
    if kind == "number":
        return numbers_equal(extract_last_number(answer), expected)
    if kind == "label":
        m = re.search(r"\b(positive|negative|neutral|mixed)\b", answer, re.IGNORECASE)
        if not m:
            return False
        got = m.group(1).lower()
        return got == expected or (expected in ("neutral", "mixed") and got in ("neutral", "mixed"))
    if kind == "contains":
        return str(expected).lower() in answer.lower()
    if kind == "entities":
        found = 0
        for ent in expected:
            name = ent.split(":")[0]
            if name.lower() in answer.lower():
                found += 1
        return found >= max(1, int(0.75 * len(expected)))  # 75% recall by name
    if kind == "code_tests":
        code = extract_code(answer)
        if not code:
            return False
        report = run_code_tests(code, expected)
        return bool(report.get("ok")) and all(r.get("passed") for r in report.get("results", []))
    return None


def main():
    load_env()
    # Usage: judge.py [input_dir] [output_dir] (defaults: test_input test_output)
    input_dir = sys.argv[1] if len(sys.argv) > 1 else "test_input"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "test_output"
    tasks = {t["task_id"]: t["prompt"]
             for t in json.load(open(os.path.join(ROOT, input_dir, "tasks.json")))}
    results = {r["task_id"]: r["answer"]
               for r in json.load(open(os.path.join(ROOT, output_dir, "results.json")))}
    expected_path = os.path.join(ROOT, "eval", "expected.json")
    expected = json.load(open(expected_path)) if os.path.exists(expected_path) else {}

    per_cat = {}
    failures = []
    for tid, prompt in tasks.items():
        cat = tid.rsplit("_", 1)[0]
        answer = results.get(tid, "")
        ok = None
        via = "llm"
        if not answer:
            ok = False
            via = "missing"
        elif tid in expected:
            ok = deterministic_check(expected[tid]["kind"], expected[tid]["answer"], answer)
            via = "det"
        if ok is None:
            ok, reason = llm_judge(prompt, answer)
            if ok is None:
                print(f"  ? {tid}: {reason}")
                continue
        per_cat.setdefault(cat, [0, 0])
        per_cat[cat][1] += 1
        per_cat[cat][0] += 1 if ok else 0
        if not ok:
            failures.append((tid, via, answer[:120].replace("\n", " ")))

    total_ok = sum(c for c, _ in per_cat.values())
    total = sum(n for _, n in per_cat.values())
    print("\nper-category accuracy:")
    for cat in sorted(per_cat):
        ok_n, n = per_cat[cat]
        print(f"  {cat:15s} {ok_n:2d}/{n:<2d} ({100 * ok_n / n:.0f}%)")
    pct = 100 * total_ok / total if total else 0
    print(f"\noverall: {total_ok}/{total} = {pct:.1f}%")
    print(f"80% gate on a 19-task sample needs >=16 correct; at {pct:.1f}% you'd "
          f"expect ~{pct / 100 * 19:.1f}/19 -> "
          f"{'PASS (with margin)' if pct >= 90 else 'PASS (thin margin)' if pct >= 84 else 'AT RISK'}")
    if failures:
        print("\nfailed tasks:")
        for tid, via, ans in failures:
            print(f"  [{via}] {tid}: {ans}")


if __name__ == "__main__":
    main()
