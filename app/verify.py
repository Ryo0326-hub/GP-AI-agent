"""Deterministic verification: safe arithmetic eval, sandboxed code execution,
synthesized test cases, logic-puzzle brute force, summarization constraints.

Everything here runs locally in pure Python and costs zero tokens.
"""
import ast
import itertools
import json
import re
import subprocess
import sys

from utils import sentence_count, word_count

# ---------------------------------------------------------------- safe eval

_ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)
_ALLOWED_CALLS = {"round", "abs", "min", "max"}


def safe_eval(expr: str):
    """Evaluate a pure-arithmetic expression. Returns a number or raises ValueError."""
    expr = (expr or "").strip().rstrip(".")
    # Strip only thousands-separator commas so round(x, 2) still parses.
    expr = re.sub(r"(?<=\d),(?=\d{3}\b)", "", expr)
    expr = expr.replace("$", "").replace("×", "*").replace("÷", "/")
    if not expr or len(expr) > 300:
        raise ValueError("empty or oversized expression")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"syntax: {e}") from e

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            v = ev(node.operand)
            return -v if isinstance(node.op, ast.USub) else v
        if isinstance(node, ast.BinOp) and isinstance(node.op, _ALLOWED_BINOPS):
            left, right = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 8:
                raise ValueError("exponent too large")
            ops = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b,
                   ast.Mult: lambda a, b: a * b, ast.Div: lambda a, b: a / b,
                   ast.FloorDiv: lambda a, b: a // b, ast.Mod: lambda a, b: a % b,
                   ast.Pow: lambda a, b: a ** b}
            return ops[type(node.op)](left, right)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id in _ALLOWED_CALLS and not node.keywords:
            args = [ev(a) for a in node.args]
            return {"round": round, "abs": abs, "min": min, "max": max}[node.func.id](*args)
        raise ValueError(f"disallowed node: {ast.dump(node)[:60]}")

    result = ev(tree)
    if not isinstance(result, (int, float)):
        raise ValueError("non-numeric result")
    return result


# ------------------------------------------------------------ code handling

_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_code(text: str) -> str:
    """Pull code out of the first fenced block, else from the first `def`."""
    m = _FENCE_RE.search(text or "")
    if m:
        return m.group(1).strip()
    m = re.search(r"(^def \w+.*)", text or "", re.MULTILINE | re.DOTALL)
    if m:
        return m.group(1).strip()
    return ""


_HARNESS = r"""
import json, sys, inspect
spec = json.load(sys.stdin)
out = {"ok": False, "defined": [], "results": []}
ns = {}
try:
    exec(compile(spec["code"], "<solution>", "exec"), ns)
except Exception as e:
    out["error"] = "exec: " + repr(e)
    print(json.dumps(out)); sys.exit(0)
funcs = {k: v for k, v in ns.items()
         if inspect.isfunction(v) and not k.startswith("_")}
out["defined"] = list(funcs)
if not funcs:
    out["error"] = "no function defined"
    print(json.dumps(out)); sys.exit(0)

def pick(name_hint):
    if name_hint and name_hint in funcs:
        return funcs[name_hint]
    return list(funcs.values())[-1]

out["ok"] = True
for t in spec.get("tests", []):
    r = {"passed": False}
    try:
        fn = pick(t.get("func"))
        got = fn(*t["args"])
        r["got"] = repr(got)[:200]
        if "expected" in t:
            exp = t["expected"]
            if isinstance(exp, float) or isinstance(got, float):
                r["passed"] = abs(float(got) - float(exp)) < 1e-6
            else:
                r["passed"] = got == exp
        else:
            r["passed"] = True  # smoke test: no exception is a pass
    except Exception as e:
        r["error"] = repr(e)[:200]
        out["ok"] = False
    out["results"].append(r)
print(json.dumps(out))
"""


def run_code_tests(code: str, tests: list, timeout: float = 5.0) -> dict:
    """Execute code + tests in an isolated subprocess. Returns the harness report."""
    payload = json.dumps({"code": code, "tests": tests})
    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-c", _HARNESS],
            input=payload, capture_output=True, text=True,
            timeout=timeout, cwd="/tmp" if sys.platform != "win32" else None,
        )
        line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
        return json.loads(line)
    except (subprocess.TimeoutExpired, json.JSONDecodeError, IndexError) as e:
        return {"ok": False, "error": f"harness: {type(e).__name__}", "results": []}


# Spec pattern -> synthesized tests. Only high-confidence, unambiguous specs.
_SPEC_TESTS = [
    (r"second[- ]largest", [
        {"args": [[1, 2, 2, 3]], "expected": 2},
        {"args": [[5, 1, 4]], "expected": 4},
        {"args": [[7, 7, 3]], "expected": 3},
    ]),
    (r"\bfactorial\b", [
        {"args": [5], "expected": 120},
        {"args": [0], "expected": 1},
    ]),
    (r"\bpalindrome\b", [
        {"args": ["racecar"], "expected": True},
        {"args": ["hello"], "expected": False},
    ]),
    (r"reverse[sd]? (a |the )?string", [
        {"args": ["abc"], "expected": "cba"},
    ]),
    (r"\b(max(imum)?|largest) (value |number |element )?(of|in|from) (a |the )?list", [
        {"args": [[3, 9, 1]], "expected": 9},
        {"args": [[-5, -2, -9]], "expected": -2},
    ]),
    (r"\b(min(imum)?|smallest) (value |number |element )?(of|in|from) (a |the )?list", [
        {"args": [[3, 9, 1]], "expected": 1},
    ]),
    (r"sum of (all )?(the )?even", [
        {"args": [[1, 2, 3, 4]], "expected": 6},
    ]),
    (r"sum of (all )?(the )?odd", [
        {"args": [[1, 2, 3, 4]], "expected": 4},
    ]),
    (r"count[s]? (the )?(number of )?vowels", [
        {"args": ["hello"], "expected": 2},
    ]),
    (r"\b(is )?prime\b", [
        {"args": [7], "expected": True},
        {"args": [8], "expected": False},
    ]),
    (r"\bfizz ?buzz\b", []),  # smoke only: output format varies
]


def synthesize_tests(prompt: str) -> list:
    p = (prompt or "").lower()
    for pattern, tests in _SPEC_TESTS:
        if re.search(pattern, p):
            return list(tests)
    return []


def generic_smoke_tests(code: str) -> list:
    """One no-crash call with a plausible argument, based on the signature."""
    m = re.search(r"def \w+\(([^)]*)\)", code or "")
    if not m:
        return []
    params = [x.strip().split(":")[0].split("=")[0].strip()
              for x in m.group(1).split(",") if x.strip()]
    params = [x for x in params if x not in ("self",)]
    if len(params) != 1:
        return []
    name = params[0].lower()
    if re.search(r"num|lst|list|arr|items|values|data|seq", name):
        return [{"args": [[3, 1, 2]]}]
    if re.search(r"s$|str|text|word|sentence", name):
        return [{"args": ["hello"]}]
    if re.search(r"n$|num|count|x$|k$", name):
        return [{"args": [5]}]
    return []


# ----------------------------------------------------------- logic puzzles

_NEG_RE = re.compile(
    r"\b([A-Z][a-z]+)\b\s+"
    r"(?:does\s*not|doesn[’']t|do\s*not|did\s*not|never|is\s*not|isn[’']t)\s+"
    r"(?:own|owns|have|has|like|likes|play|plays|drink|drinks|wear|wears|got|choose|pick)?\s*"
    r"(?:the |a |an )?(\w+)")
_POS_RE = re.compile(
    r"\b([A-Z][a-z]+)\b (?:owns|has|likes|plays|drinks|wears|got|chose|picked)\s*"
    r"(?:the |a |an )?(\w+)")
_QUESTION_RE = re.compile(
    r"\bwho (?:owns|has|likes|plays|drinks|wears|got|chose|picked)\s*(?:the |a |an )?(\w+)")


def solve_assignment_puzzle(prompt: str):
    """Brute-force 'N people, N different items' puzzles.

    Returns (person, item, assignment_dict) when exactly one assignment
    satisfies all parsed constraints and answers the question, else None.
    """
    text = prompt or ""
    # Items: after 'different <noun>:' e.g. "a different pet: cat, dog, bird."
    m = re.search(r"different \w+s?\s*[:\-]\s*([^.?!]+)", text, re.IGNORECASE)
    if not m:
        return None
    items = [w.strip().lower() for w in re.split(r",| and | or ", m.group(1)) if w.strip()]
    items = [re.sub(r"^(a|an|the) ", "", i) for i in items if 0 < len(i.split()) <= 2]
    # People: capitalized names appearing before 'each'.
    head = text.split("each")[0]
    names = re.findall(r"\b([A-Z][a-z]+)\b", head)
    stop = {"Three", "Four", "Five", "Two", "Six", "The", "A", "An", "If", "Who"}
    names = [n for n in dict.fromkeys(names) if n not in stop]
    if not (2 <= len(items) <= 6) or len(names) != len(items):
        return None

    positives = [(a, b.lower()) for a, b in _POS_RE.findall(text)
                 if a in names and b.lower() in items]
    negatives = [(a, b.lower()) for a, b in _NEG_RE.findall(text)
                 if a in names and b.lower() in items]
    qm = _QUESTION_RE.search(text.lower())
    if not qm or qm.group(1) not in items or not (positives or negatives):
        return None
    target_item = qm.group(1)

    solutions = []
    for perm in itertools.permutations(items):
        assign = dict(zip(names, perm))
        if all(assign[a] == b for a, b in positives) and \
           all(assign[a] != b for a, b in negatives):
            solutions.append(assign)
    if len(solutions) != 1:
        return None
    assign = solutions[0]
    person = next(n for n, it in assign.items() if it == target_item)
    return person, target_item, assign


# ----------------------------------------------- summarization constraints

def parse_summary_constraints(prompt: str) -> dict:
    p = (prompt or "").lower()
    c = {}
    m = re.search(r"exactly (one|two|three|1|2|3) sentences?", p)
    if m:
        c["sentences_exact"] = {"one": 1, "two": 2, "three": 3}.get(m.group(1)) or int(m.group(1))
    elif re.search(r"\bin (a|one) (single )?sentence\b|\bone-sentence\b|\bsingle sentence\b", p):
        c["sentences_exact"] = 1
    m = re.search(r"(?:no more than|at most|maximum(?: of)?|under|fewer than|less than|within|in) (\d+) words", p)
    if m:
        limit = int(m.group(1))
        if re.search(r"fewer than|less than|under", m.group(0)):
            limit -= 1
        c["words_max"] = limit
    m = re.search(r"exactly (\d+) words", p)
    if m:
        c["words_exact"] = int(m.group(1))
    return c


def check_summary(answer: str, constraints: dict):
    """Return (ok, list_of_violations)."""
    violations = []
    if "sentences_exact" in constraints:
        n = sentence_count(answer)
        if n != constraints["sentences_exact"]:
            violations.append(f"expected {constraints['sentences_exact']} sentence(s), got {n}")
    if "words_max" in constraints and word_count(answer) > constraints["words_max"]:
        violations.append(f"over {constraints['words_max']} words ({word_count(answer)})")
    if "words_exact" in constraints and word_count(answer) != constraints["words_exact"]:
        violations.append(f"expected exactly {constraints['words_exact']} words")
    return (not violations), violations


# --------------------------------------------------------- generic checks

_HEDGE_RE = re.compile(
    r"\bas an ai\b|\bi (do not|don't) know\b|\bi cannot\b|\bi'm not sure\b|\bunable to\b",
    re.IGNORECASE)


def looks_confident(answer: str, min_len: int = 15) -> bool:
    a = (answer or "").strip()
    return len(a) >= min_len and not _HEDGE_RE.search(a)
