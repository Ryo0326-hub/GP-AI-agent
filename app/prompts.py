"""Per-category system prompts and generation caps for the local model.

Caps are in completion tokens. They bound both latency (2 vCPU CPU inference)
and answer length; answers must still fully address multi-part questions,
so factual/math prompts explicitly demand completeness.
"""

MAX_TOKENS = {
    "sentiment": 60,
    "ner": 120,
    "math": 200,           # brief reasoning + Expression + Answer lines
    "summarization": 80,
    "factual": 110,
    "logic": 200,
    "debug": 400,
    "codegen": 400,
}

# Escalation caps: raw prompt to the Fireworks model, so allow a bit more room.
ESCALATION_MAX_TOKENS = {
    "sentiment": 200,
    "ner": 400,
    "math": 300,
    "summarization": 500,
    "factual": 400,
    "logic": 300,
    "debug": 500,
    "codegen": 500,
}

SYSTEM = {
    "factual": (
        "Answer accurately and completely in 1-3 sentences. If the question has "
        "multiple parts, answer every part. No preamble, no hedging."
    ),
    "math": (
        "Solve the problem with brief step-by-step arithmetic. Then end with "
        "exactly two lines:\n"
        "Expression: <one arithmetic expression using only numbers and + - * / ( ) that evaluates to the answer>\n"
        "Answer: <the final number, with unit if relevant>"
    ),
    "sentiment": (
        "Classify the sentiment as Positive, Negative, Neutral, or Mixed. "
        "Reply with the label first, then a one-sentence justification citing the text."
    ),
    "summarization": (
        "Summarize exactly as instructed. Obey every format and length constraint "
        "literally (e.g. 'exactly one sentence' means one sentence, no more). "
        "Output only the summary."
    ),
    "ner": (
        "Extract every named entity from the text with its type "
        "(person, organization, location, or date). Output one entity per line as:\n"
        "Entity - type\n"
        "List all of them; do not add commentary."
    ),
    "debug": (
        "Identify the bug in one sentence, then provide the fully corrected code "
        "in a ```python code block. The corrected code must be complete and runnable."
    ),
    "codegen": (
        "Begin immediately with ```python and write the requested function in one code block. "
        "It must be correct, handle the edge cases mentioned, and use no external "
        "libraries unless asked. Do not restate the task or include examples."
    ),
    "logic": (
        "Solve the puzzle by brief step-by-step deduction. "
        "End with exactly one line:\n"
        "Answer: <the answer>"
    ),
}

# Prompt used for a single math retry when the expression check fails.
MATH_RETRY_SYSTEM = (
    "Solve carefully, one small arithmetic step per line, checking each step. "
    "Then end with exactly two lines:\n"
    "Expression: <one arithmetic expression using only numbers and + - * / ( ) that evaluates to the answer>\n"
    "Answer: <the final number, with unit if relevant>"
)

# Prompt used for one local repair round when generated code fails its tests.
CODE_REPAIR_SYSTEM = (
    "Write or fix the requested code. Begin immediately with ```python. Reply with "
    "only one corrected, complete code block, nothing else."
)
