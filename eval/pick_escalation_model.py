#!/usr/bin/env python3
"""Dev-only: pick the escalation tier empirically, not by name.

For every selected model in .env ALLOWED_MODELS, runs ~20 representative hard
tasks (the kind the router will actually escalate) with exact candidate
runtime message shapes: either the raw user prompt or the category-specific
system prompt from app/prompts.py. It reports accuracy and
tokens-per-correct-answer from the API's billed usage totals.

Reasoning-style models can burn hidden thinking tokens, so each model is also
evaluated with OpenAI-compatible reasoning switches (--modes); a mode is
skipped for a model if the endpoint rejects it.

The winner is NOT hardcoded into the agent. Bake its name pattern via the
PREFERRED_MODEL_HINTS build arg (see Dockerfile), which the runtime matches
against whatever ALLOWED_MODELS the judge injects.

Usage: python3 eval/pick_escalation_model.py \
  [--modes default,reasoning_effort=none] [--styles raw,category]
"""
import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "app"))
sys.path.insert(0, os.path.join(ROOT, "eval"))

import requests  # noqa: E402

from label_local import grade  # noqa: E402
from prompts import ESCALATION_MAX_TOKENS, SYSTEM  # noqa: E402
from classifier import classify  # noqa: E402

SAFE_ACCURACY_FLOOR = 0.90
PROMPT_STYLES = ("raw", "category")

# ~20 hard tasks with deterministic ground truth, in the style the escalated
# share of a real eval looks like: multi-step math, dense logic, algorithmic
# codegen, subtle debugging, ambiguous NER, backhanded sentiment, strict
# summarization, multi-part factual.
HARD_TASKS = [
    ("A tank starts with 480 liters. It drains at 8 liters per minute for 15 "
     "minutes, then is refilled at 12 liters per minute for 20 minutes, then "
     "drains again at 5 liters per minute for 10 minutes. How many liters are "
     "in the tank now?", 550, "number"),
    ("A price is increased by 20%, then decreased by 20%, then increased by "
     "10%. If the original price was $200, what is the final price in dollars?",
     211.20, "number"),
    ("You invest $1000. In year 1 it grows by 5%, in year 2 by 8%, in year 3 "
     "it shrinks by 3%. What is the final amount in dollars, rounded to 2 "
     "decimal places?", 1099.98, "number"),
    ("Pipe A can fill a pool in 6 hours. Pipe B can fill the same pool in 4 "
     "hours. If both pipes are opened together, how many hours will it take "
     "to fill the pool? Answer as a decimal rounded to 1 decimal place.",
     2.4, "number"),
    ("A farmer has chickens and rabbits, 23 heads and 62 legs in total. "
     "How many rabbits are there?", 8, "number"),
    ("Five boxes, P, Q, R, S, T, contain weights of 2, 4, 6, 8, and 10 kg, "
     "one weight per box. Box P and box Q together weigh 6 kg. Box R is "
     "heavier than box S. Box T is the heaviest of all five boxes. Box S and "
     "box Q differ in weight by exactly 2 kg. What is the weight of box R, "
     "in kg?", "8", "contains"),
    ("Four kids, Max, Nia, Omar, and Pia, each play a different sport: "
     "soccer, tennis, chess, golf. Max plays chess. Nia does not play soccer "
     "and does not play golf. Omar plays golf. Who plays soccer?",
     "Pia", "contains"),
    ("In a race, Uma finished before Vic, and Vic finished before Wes. Tia "
     "finished after Uma but before Vic. Who finished second?",
     "Tia", "contains"),
    ("If all roses are flowers and some flowers fade quickly, can we conclude "
     "that some roses fade quickly? Answer yes or no and explain briefly.",
     "no", "contains"),
    ("Write a Python function edit_distance(a, b) that returns the minimum "
     "number of single-character insertions, deletions, or substitutions "
     "required to transform string a into string b.",
     [{"args": ["kitten", "sitting"], "expected": 3},
      {"args": ["", "abc"], "expected": 3}], "code_tests"),
    ("Write a Python function that returns the length of the longest strictly "
     "increasing subsequence in a list of integers.",
     [{"args": [[10, 9, 2, 5, 3, 7, 101, 18]], "expected": 4},
      {"args": [[5, 4, 3, 2, 1]], "expected": 1}], "code_tests"),
    ("Write a Python function that checks whether a string of brackets ()[]{} "
     "is balanced and properly nested.",
     [{"args": ["([]{})"], "expected": True},
      {"args": ["([)]"], "expected": False}], "code_tests"),
    ("This function should compute the running total of a list but has a "
     "bug: def running(nums):\n    out = []\n    t = 0\n    for n in nums:\n"
     "        out.append(t)\n        t += n\n    return out\nFind and fix it.",
     [{"args": [[1, 2, 3]], "expected": [1, 3, 6]}], "code_tests"),
    ("This function should check whether a number is prime but has a bug: "
     "def is_prime(n): return all(n % i != 0 for i in range(2, n)). "
     "It wrongly says 1 is prime. Find and fix it.",
     [{"args": [7], "expected": True}, {"args": [1], "expected": False},
      {"args": [2], "expected": True}], "code_tests"),
    ("Extract all named entities and their types (person, organization, "
     "location, date) from: Amazon announced that Jordan, the new VP hired "
     "from Washington, will lead the Phoenix office starting in April.",
     ["Jordan:person", "Amazon:organization", "Washington:location",
      "Phoenix:location", "April:date"], "entities"),
    ("Extract all named entities and their types (person, organization, "
     "location, date) from: Turner joined Sterling Bank in Sterling, "
     "Colorado, replacing Bell who moved to a Bell Labs research role in June.",
     ["Turner:person", "Sterling Bank:organization", "Bell Labs:organization",
      "Colorado:location", "June:date"], "entities"),
    ("Classify the sentiment of this review: I wouldn't say the food was bad, "
     "exactly, but I also wouldn't rush back.", "negative", "label"),
    ("Classify the sentiment of this review: Well, at least it wasn't the "
     "WORST customer service I've ever had.", "negative", "label"),
    ("Summarize the following paragraph in exactly one sentence: The startup "
     "raised $4.2 million in seed funding and grew its user base to 50,000 "
     "monthly active users within six months, driven mainly by a viral "
     "referral program that cut acquisition costs by half.",
     {"keyterms": ["4.2", "50,000"]}, "summary"),
    ("Which country hosted the 2016 Summer Olympics, and in which city?",
     ["brazil", "rio"], "contains_all"),
]


def load_env():
    path = os.path.join(ROOT, ".env")
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def call(base, key, model, prompt, max_tokens, extra, system=None):
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    body = {"model": model, "temperature": 0, "max_tokens": max_tokens,
            "messages": messages}
    body.update(extra)
    try:
        r = requests.post(base + "/chat/completions", timeout=90,
                          headers={"Authorization": f"Bearer {key}"}, json=body)
    except requests.RequestException as e:
        return None, 0, f"network error: {type(e).__name__}"
    if not r.ok:
        return None, 0, f"HTTP {r.status_code}: {(r.text or '')[:120]}"
    data = r.json()
    msg = data["choices"][0]["message"]
    text = (msg.get("content") or "").strip()
    tokens = int((data.get("usage") or {}).get("total_tokens") or 0)
    return text, tokens, None


def parse_modes(spec):
    modes = []
    for m in spec.split(","):
        m = m.strip()
        if m == "default":
            modes.append(("default", {}))
        elif "=" in m:
            k, v = m.split("=", 1)
            modes.append((m, {k: v}))
    return modes


def parse_styles(spec):
    styles = [s.strip().lower() for s in spec.split(",") if s.strip()]
    invalid = [s for s in styles if s not in PROMPT_STYLES]
    if not styles or invalid:
        raise ValueError(
            f"styles must be a comma-separated subset of {PROMPT_STYLES}; "
            f"invalid={invalid}")
    return styles


def select_models(allowed, spec):
    """Filter ALLOWED_MODELS by exact full ID or exact final path segment."""
    if not spec:
        return allowed
    wanted = {s.strip() for s in spec.split(",") if s.strip()}
    selected = [m for m in allowed
                if m in wanted or m.rsplit("/", 1)[-1] in wanted]
    missing = wanted - {m for m in selected} - {
        m.rsplit("/", 1)[-1] for m in selected}
    if missing:
        raise ValueError(f"requested models not found in ALLOWED_MODELS: "
                         f"{sorted(missing)}")
    return selected


def system_for_style(prompt, prompt_style):
    if prompt_style == "raw":
        return None
    if prompt_style == "category":
        return SYSTEM.get(classify(prompt))
    raise ValueError(f"unknown prompt style: {prompt_style}")


def eval_model(base, key, model, mode_name, extra, prompt_style="raw"):
    def one(task):
        prompt, expected, kind = task
        cap = ESCALATION_MAX_TOKENS.get(classify(prompt), 300)
        system = system_for_style(prompt, prompt_style)
        text, tokens, err = call(base, key, model, prompt, cap, extra, system)
        if err:
            return None, 0, err
        return grade(kind, expected, text, prompt), tokens, None

    # Probe with the first task so an unsupported mode does not burn the other
    # 19 calls. A failed probe is recorded as a failed run (not removed from
    # consideration); the unattempted answers are conservatively counted as
    # failures too.
    ok0, tok0, err0 = one(HARD_TASKS[0])
    if err0:
        return {"model": model, "mode": mode_name, "style": prompt_style,
                "n": len(HARD_TASKS),
                "correct": 0, "accuracy": 0.0, "total_tokens": 0,
                "tokens_per_correct": 0.0,
                "call_errors": 1, "tasks_not_run": len(HARD_TASKS) - 1,
                "probe_error": err0}
    correct, tokens, errors = int(bool(ok0)), tok0, 0
    with ThreadPoolExecutor(max_workers=4) as ex:
        for ok, tok, err in ex.map(one, HARD_TASKS[1:]):
            if err:
                errors += 1
                continue
            correct += bool(ok)
            tokens += tok
    # Transport/API failures are wrong answers at judge time, not samples that
    # disappear from the denominator. Keeping n fixed prevents a flaky model
    # from looking artificially accurate and cheap.
    n = len(HARD_TASKS)
    return {"model": model, "mode": mode_name, "style": prompt_style,
            "n": n, "correct": correct,
            "accuracy": round(correct / max(1, n), 3), "total_tokens": tokens,
            "tokens_per_correct": round(tokens / max(1, correct), 1),
            "call_errors": errors}


def rank_results(results, min_accuracy=SAFE_ACCURACY_FLOOR):
    """Rank viable escalation choices without trading away the accuracy gate.

    Error-free models at or above ``min_accuracy`` form the safe tier. Within
    that tier, accuracy is the primary key and token efficiency breaks ties.
    If no result clears the tier, the highest observed accuracy wins and call
    errors/token cost only break ties; callers should surface that fallback as
    unsafe rather than silently calling it a winner.
    """
    scored = [r for r in results if "error" not in r and r.get("n", 0) > 0]

    def safe(r):
        return r.get("call_errors", 0) == 0 and r.get("accuracy", 0) >= min_accuracy

    return sorted(scored, key=lambda r: (
        not safe(r),
        -r.get("accuracy", 0),
        r.get("call_errors", 0),
        r.get("tokens_per_correct", float("inf")),
        r.get("total_tokens", float("inf")),
    ))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", default="default,reasoning_effort=none",
                    help="comma list: default and/or key=value extra params")
    ap.add_argument("--styles", default="raw,category",
                    help="comma list of exact message styles: raw,category")
    ap.add_argument("--models", default="",
                    help="optional comma list of exact allowed model IDs or "
                         "model-name suffixes")
    ap.add_argument("--out", default=os.path.join(ROOT, "data",
                                                  "escalation_model_report.json"))
    ap.add_argument("--min-accuracy", type=float, default=SAFE_ACCURACY_FLOOR,
                    help="minimum error-free accuracy required before token "
                         "efficiency can select the winner")
    args = ap.parse_args()
    load_env()
    base = (os.environ.get("FIREWORKS_BASE_URL") or "").rstrip("/")
    key = os.environ.get("FIREWORKS_API_KEY") or ""
    allowed = [m.strip() for m in (os.environ.get("ALLOWED_MODELS") or "").split(",")
               if m.strip()]
    if not (base and key and allowed):
        print("need FIREWORKS_BASE_URL / FIREWORKS_API_KEY / ALLOWED_MODELS in .env")
        return 1
    try:
        styles = parse_styles(args.styles)
        allowed = select_models(allowed, args.models)
    except ValueError as e:
        ap.error(str(e))

    results = []
    for model in allowed:
        for mode_name, extra in parse_modes(args.modes):
            for prompt_style in styles:
                print(f"evaluating {model} [{mode_name}] style={prompt_style} "
                      f"on {len(HARD_TASKS)} hard tasks...", flush=True)
                res = eval_model(base, key, model, mode_name, extra, prompt_style)
                results.append(res)
                if "error" in res:
                    print(f"  skipped: {res['error']}")
                else:
                    print(f"  accuracy={res['accuracy']} "
                          f"tokens={res['total_tokens']} "
                          f"tokens/correct={res['tokens_per_correct']} "
                          f"errors={res['call_errors']}")

    scored = rank_results(results, args.min_accuracy)
    print(f"\n{'model':55s} {'mode':22s} {'style':>9s} "
          f"{'acc':>5s} {'tokens':>7s} {'tok/ok':>7s}")
    for r in scored:
        print(f"{r['model']:55s} {r['mode']:22s} {r['style']:>9s} "
              f"{r['accuracy']:5.0%} {r['total_tokens']:7d} "
              f"{r['tokens_per_correct']:7.1f}")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    if scored:
        best = scored[0]
        safe = (best["call_errors"] == 0
                and best["accuracy"] >= args.min_accuracy)
        if not safe:
            print(f"\nWARNING: no error-free model cleared the "
                  f"{args.min_accuracy:.0%} accuracy floor; best observed "
                  "fallback follows, but do not bake it without more testing.")
        label = "winner" if safe else "fallback"
        print(f"\n{label}: {best['model']} [{best['mode']}] "
              f"style={best['style']}")
        if safe:
            hint = best["model"].rsplit("/", 1)[-1]
            print(f"bake it in with:  --build-arg "
                  f"PREFERRED_MODEL_HINTS=\"{hint}\"")
            if best["mode"] != "default":
                k, v = best["mode"].split("=", 1)
                print(f"and:  --build-arg FIREWORKS_EXTRA_BODY="
                      f"'{json.dumps({k: v})}'")
            print(f"runtime prompt style: {best['style']}")
            if best["style"] == "category":
                print("runtime change before build: pass "
                      "SYSTEM[classify(prompt)] as the Fireworks system message")
            else:
                print("runtime change before build: leave the Fireworks system "
                      "message unset")
            print(f"also recalibrate the threshold tuner: retrain with "
                  f"--esc-tokens "
                  f"{max(1, best['total_tokens'] // len(HARD_TASKS))}")
        else:
            print("no build arguments emitted: collect a reliable passing run first")
    print(f"\nfull report: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
