import main


class ScriptedLM:
    def __init__(self, answers):
        self.answers = iter(answers)
        self.calls = 0

    def chat(self, *args, **kwargs):
        self.calls += 1
        return next(self.answers)


def test_mixed_sentiment_normalizes_self_contradictory_label():
    lm = ScriptedLM([
        "Negative\nThe review contains both positive praise and negative criticism."
    ])
    answer, verified = main.solve_simple(lm, "review", "sentiment", 20)
    assert answer.startswith("Mixed")
    assert verified


def test_codegen_retries_when_first_attempt_has_no_code(monkeypatch):
    lm = ScriptedLM([
        "Here is the requested function:",
        "```python\ndef second_largest(nums):\n    return sorted(set(nums))[-2]\n```",
    ])
    times = iter([0.0, 1.0])
    monkeypatch.setattr(main.time, "monotonic", lambda: next(times))
    answer, verified = main.solve_code(
        lm,
        "Write a Python function that returns the second-largest number in a list, "
        "handling duplicates correctly.",
        "codegen",
        20,
    )
    assert "def second_largest" in answer
    assert verified
    assert lm.calls == 2


def test_ner_rejects_a_truncated_entry():
    lm = ScriptedLM(["Maria Sanchez - person\nFireworks"])
    _, verified = main.solve_simple(lm, "extract entities", "ner", 20)
    assert not verified


def test_ner_accepts_only_complete_typed_lines():
    lm = ScriptedLM(["Maria Sanchez - person\nFireworks AI - organization"])
    _, verified = main.solve_simple(lm, "extract entities", "ner", 20)
    assert verified
