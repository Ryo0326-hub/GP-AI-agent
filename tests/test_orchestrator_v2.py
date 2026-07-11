"""End-to-end orchestrator paths with a stubbed router, LM, and Fireworks:
immediate planned escalation, router-off fallback, trust_local acceptance,
and escalation failure falling back to the kept local answer."""
import importlib
import json
import sys
import types


class StubLM:
    def __init__(self, path):
        self.cutoff_count = 0
        self.tok_per_sec = 10.0

    def benchmark(self):
        return 10.0

    def chat(self, system, user, max_tokens, time_budget=30.0):
        if "Expression:" in (system or ""):
            return "Step.\nExpression: 2+2\nAnswer: 4"
        if "sentiment" in (system or "").lower():
            return "Positive. The reviewer clearly loves it."
        return "hm"  # too short -> unverified for factual


class StubFW:
    def __init__(self, answer="ESCALATED ANSWER"):
        self.answer = answer
        self.calls = []
        self.systems = []
        self.available = True
        self.model = "stub-model"
        self.allowed = ["stub-model"]
        self.calls_attempted = 0
        self.calls_succeeded = 0
        self.total_tokens = 0

    def complete(self, prompt, cap, timeout=25.0, system=None):
        self.calls.append(prompt)
        self.systems.append(system)
        self.calls_attempted += 1
        if self.answer:
            self.calls_succeeded += 1
            self.total_tokens += 50
        return self.answer


TASKS = [
    {"task_id": "f1", "prompt": "What is the capital of Australia?"},
    {"task_id": "m1", "prompt": "What is 2+2? Calculate the total."},
    {"task_id": "s1", "prompt": "Classify the sentiment of this review: great!"},
]


def _run(monkeypatch, tmp_path, scores, config, fw):
    inp = tmp_path / "tasks.json"
    out = tmp_path / "results.json"
    inp.write_text(json.dumps(TASKS))
    monkeypatch.setenv("INPUT_PATH", str(inp))
    monkeypatch.setenv("OUTPUT_PATH", str(out))
    monkeypatch.setenv("TOTAL_DEADLINE_S", "540")
    monkeypatch.delenv("ROUTER_THRESHOLD", raising=False)
    for var in ("FIREWORKS_API_KEY", "FIREWORKS_BASE_URL", "ALLOWED_MODELS"):
        monkeypatch.delenv(var, raising=False)

    stub = types.ModuleType("local_model")
    stub.LocalLM = StubLM
    monkeypatch.setitem(sys.modules, "local_model", stub)

    import main
    main = importlib.reload(main)
    monkeypatch.setattr(main, "score_and_free", lambda prompts: scores)
    monkeypatch.setattr(main, "load_config", lambda: config)
    monkeypatch.setattr(main, "Fireworks", lambda: fw)
    assert main.main() == 0
    results = {r["task_id"]: r["answer"] for r in json.loads(out.read_text())}
    return main, results


def test_planned_escalation_dispatches_immediately(monkeypatch, tmp_path):
    fw = StubFW()
    config = {"threshold": 0.5, "category_policy": {},
              "expected_completion_tokens": {}}
    main, results = _run(monkeypatch, tmp_path,
                         scores=[0.95, 0.05, 0.05], config=config, fw=fw)
    stats = main.LAST_RUN_STATS
    assert stats["planned_escalations"] == 1
    assert stats["router_scored"] is True
    assert fw.calls == [TASKS[0]["prompt"]]  # only the predicted-fail task
    assert fw.systems == [main.SYSTEM["factual"]]
    assert results["f1"] == "ESCALATED ANSWER"
    assert stats["paths"] == {main.ESCALATED: 1, main.LOCAL_VERIFIED: 2}


def test_router_unavailable_falls_back_to_verification_gating(monkeypatch, tmp_path):
    fw = StubFW()
    main, results = _run(monkeypatch, tmp_path,
                         scores=None, config=None, fw=fw)
    stats = main.LAST_RUN_STATS
    assert stats["planned_escalations"] == 0
    assert stats["router_scored"] is False
    # The unverified factual answer escalates reactively; verified ones don't.
    assert fw.calls == [TASKS[0]["prompt"]]
    assert fw.systems == [main.SYSTEM["factual"]]
    assert results["f1"] == "ESCALATED ANSWER"
    assert stats["paths"] == {main.ESCALATED: 1, main.LOCAL_VERIFIED: 2}


def test_trust_local_accepts_unverified_without_tokens(monkeypatch, tmp_path):
    fw = StubFW()
    config = {"threshold": 0.5,
              "category_policy": {"trust_local": ["factual"]},
              "expected_completion_tokens": {}}
    main, results = _run(monkeypatch, tmp_path,
                         scores=[0.05, 0.05, 0.05], config=config, fw=fw)
    stats = main.LAST_RUN_STATS
    assert fw.calls == []  # zero Fireworks traffic
    assert results["f1"] == "hm"
    assert stats["paths"] == {main.LOCAL_UNVERIFIED: 1, main.LOCAL_VERIFIED: 2}


def test_failed_escalation_falls_back_to_local_answer(monkeypatch, tmp_path):
    fw = StubFW(answer="")  # Fireworks up but returning nothing usable
    config = {"threshold": 0.5, "category_policy": {},
              "expected_completion_tokens": {}}
    main, results = _run(monkeypatch, tmp_path,
                         scores=[0.95, 0.05, 0.05], config=config, fw=fw)
    stats = main.LAST_RUN_STATS
    # The planned escalation failed; the endgame answers it locally instead
    # of emitting the fallback string.
    assert results["f1"] == "hm"
    assert stats["paths"].get(main.FALLBACK) is None
    assert all(a.strip() for a in results.values())


def test_always_escalate_category_ignores_low_score(monkeypatch, tmp_path):
    fw = StubFW()
    config = {"threshold": 0.5,
              "category_policy": {"always_escalate": ["factual"]},
              "expected_completion_tokens": {}}
    main, results = _run(monkeypatch, tmp_path,
                         scores=[0.01, 0.01, 0.01], config=config, fw=fw)
    assert main.LAST_RUN_STATS["planned_escalations"] == 1
    assert results["f1"] == "ESCALATED ANSWER"


def test_slow_local_model_is_retained_as_provider_failure_fallback(
        monkeypatch, tmp_path):
    monkeypatch.setattr(StubLM, "benchmark", lambda self: 2.0)
    fw = StubFW(answer="")
    main, results = _run(
        monkeypatch, tmp_path, scores=None, config=None, fw=fw)

    assert all(answer.strip() for answer in results.values())
    assert results["f1"] == "hm"
    assert main.LAST_RUN_STATS["paths"].get(main.FALLBACK) is None
