"""Remote-answer verification, retry, and the factual consensus gate."""
import sys
import types

sys.modules.setdefault("local_model", types.ModuleType("local_model"))

import main  # noqa: E402


def test_remote_verified_math_accepts_consistent_expression():
    text = "Step.\nExpression: 240 - 240*0.10 - 45\nAnswer: 171"
    assert main._remote_verified("math", "p", text)


def test_remote_verified_math_rejects_mismatch():
    text = "Step.\nExpression: 2+2\nAnswer: 5"
    assert not main._remote_verified("math", "p", text)


def test_remote_verified_summary_checks_constraints():
    prompt = "Summarize in exactly one sentence: stuff."
    assert main._remote_verified("summarization", prompt, "One sentence only.")
    assert not main._remote_verified("summarization", prompt,
                                     "Two sentences. Right here.")


def test_answers_agree_same_fact_different_verbosity():
    a = "The capital is Canberra, near Lake Burley Griffin."
    b = ("The capital of Australia is Canberra, and it is located near "
         "Lake Burley Griffin, an artificial lake.")
    assert main._answers_agree(a, b)


def test_answers_disagree_on_named_fact():
    a = "The capital of Australia is Canberra, and it is near the Tasman Sea."
    b = ("The capital of Australia is Canberra, and it is located near "
         "Lake Burley Griffin, an artificial lake.")
    assert not main._answers_agree(a, b)


def test_answers_disagree_on_numbers():
    assert not main._answers_agree("The tower is 300 m tall.",
                                   "The tower is 324 m tall.")


class ConsensusFW:
    def __init__(self, answers, arbiter_reply="B"):
        self.answers = dict(answers)
        self.arbiter_reply = arbiter_reply
        self.model = "flash"
        self.allowed = ["flash", "pro", "kimi"]
        self.calls = []

    def pick_distinct(self, exclude, hints):
        for m in self.allowed:
            if m not in exclude:
                return m
        return None

    def secondary_model(self):
        return "pro"

    def complete(self, prompt, cap, timeout=25.0, system=None,
                 model_override=None):
        model = model_override or self.model
        self.calls.append(model)
        if system == main._ARBITER_SYSTEM:
            return self.arbiter_reply
        return self.answers.get(model, "")


def test_factual_consensus_disagreement_arbitrated():
    fw = ConsensusFW({"flash": "Canberra is near the Tasman Sea.",
                      "pro": "Canberra is near Lake Burley Griffin."},
                     arbiter_reply="B")
    out = main._factual_consensus(fw, "capital?", 100, "sys", 20.0)
    assert out == "Canberra is near Lake Burley Griffin."
    assert fw.calls == ["flash", "pro", "kimi"]


def test_factual_consensus_agreement_skips_arbiter():
    fw = ConsensusFW({"flash": "Paris, on the Seine.",
                      "pro": "Paris, located on the Seine."})
    out = main._factual_consensus(fw, "capital?", 100, "sys", 20.0)
    assert out == "Paris, on the Seine."
    assert fw.calls == ["flash", "pro"]


def test_factual_consensus_secondary_failure_keeps_primary():
    fw = ConsensusFW({"flash": "Canberra.", "pro": ""})
    out = main._factual_consensus(fw, "capital?", 100, "sys", 20.0)
    assert out == "Canberra."
