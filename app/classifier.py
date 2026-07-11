"""Heuristic task-category classifier. Pure Python, zero LLM calls.

Categories: sentiment, summarization, ner, debug, codegen, math, logic, factual.
Rule order matters: the most distinctive surface cues are checked first.
"""
import re

CATEGORIES = ("sentiment", "summarization", "ner", "debug", "codegen",
              "math", "logic", "factual")

_CODE_HINT = re.compile(r"```|\bdef \w+|\bfunction\b|\breturn\b|\bclass \w+\(|=>|\bprintln\b|\bconsole\.log\b")
_DEBUG = re.compile(r"\bbug(s|gy)?\b|\bdebug\b|\bfix(es|ed|ing)?\b|\bincorrect(ly)?\b|\bdoesn'?t work\b|\bnot work(ing)?\b|\berror\b|\bwrong (output|result|answer)\b")
_CODEGEN = re.compile(
    r"\b(write|create|implement|develop|code|build)\b[^.?!]{0,60}\b(function|program|script|method|class|snippet|code)\b"
    r"|\bfunction that\b|\bprogram that\b")
_MATH_VERB = re.compile(
    r"\bhow (many|much|far|long|old)\b|\bcalculate\b|\bcompute\b|\bwhat is \d|\bwhat'?s \d"
    r"|\bpercent(age)?\b|%|\bsum\b|\bproduct of\b|\bdifference\b|\bremain(s|ing)?\b|\bleft over\b"
    r"|\btotal\b|\baverage\b|\bper (hour|day|week|month|item|unit)\b|\bcost(s)?\b|\bprice\b"
    r"|\bdiscount\b|\binterest\b|\bspeed\b|\bdistance\b|\barea\b|\bperimeter\b|\bvolume\b")
_LOGIC = re.compile(
    r"\beach (own|owns|has|have|like|likes|wear|wears|drink|drinks|play|plays)\b"
    r"|\bdifferent (pet|color|colour|car|house|drink|sport|hobby|job|instrument|fruit)\b"
    r"|\bwho (owns|has|likes|plays|wears|drinks|lives|sits)\b"
    r"|\balways (lies|tells the truth)\b|\bliar\b|\btruth-?teller\b"
    r"|\bif and only if\b|\bimplies\b|\bdeduce\b|\blogic puzzle\b"
    r"|\b(sits|seated|stands) (next to|between|left of|right of)\b"
    r"|\bcan we conclude\b|\bdoes it follow\b|\bnecessarily true\b"
    r"|\bfinished (before|after)\b|\bwho (finished|came|arrived)\b"
    r"|\btaller than\b|\bolder than\b|\bfaster than\b.*\bwho\b")


def classify(prompt: str) -> str:
    p = (prompt or "").lower()
    has_code = bool(_CODE_HINT.search(prompt or ""))

    if re.search(r"\bsentiment\b", p) or re.search(
            r"\bclassify\b[^.?!]{0,80}\b(review|tweet|comment|feedback|text|statement)\b", p):
        return "sentiment"
    if re.search(r"\bsummar(y|ize|ise|ies|izing|ising|ization|isation)\b|\btl;?dr\b|\bcondense\b", p):
        return "summarization"
    if re.search(r"\bentit(y|ies)\b|\bnamed entit", p) or re.search(
            r"\bextract\b[^.?!]{0,80}\b(people|persons?|organi[sz]ations?|locations?|dates?|names)\b", p):
        return "ner"
    if has_code and _DEBUG.search(p):
        return "debug"
    if _CODEGEN.search(p) or (has_code and re.search(r"\bwrite\b|\bimplement\b|\bcomplete\b", p)):
        return "codegen"
    if _LOGIC.search(p):
        return "logic"
    if re.search(r"\d", p) and _MATH_VERB.search(p):
        return "math"
    # Numberless arithmetic phrasing ("twice as many...") or spelled-out numbers.
    if _MATH_VERB.search(p) and re.search(
            r"\b(twice|half|double|triple|one|two|three|four|five|six|seven|eight|nine|ten|dozen)\b", p):
        return "math"
    return "factual"
