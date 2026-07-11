"""End-of-run diagnostics must be truthful: full category histogram, per-path
counts, fireworks attempted vs succeeded. Runs main.main() with a stubbed
local model and no Fireworks env."""
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


def test_stats_cover_all_tasks_and_paths(monkeypatch, tmp_path):
    tasks = [
        {"task_id": "m1", "prompt": "What is 2+2? Calculate the total."},
        {"task_id": "s1", "prompt": "Classify the sentiment of this review: great!"},
        {"task_id": "f1", "prompt": "What is the capital of Australia?"},
    ]
    inp = tmp_path / "tasks.json"
    out = tmp_path / "results.json"
    inp.write_text(json.dumps(tasks))
    monkeypatch.setenv("INPUT_PATH", str(inp))
    monkeypatch.setenv("OUTPUT_PATH", str(out))
    monkeypatch.setenv("TOTAL_DEADLINE_S", "540")
    monkeypatch.setenv("PER_TASK_CAP_S", "25")
    monkeypatch.setenv("PER_TASK_MIN_S", "20")
    monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)
    monkeypatch.delenv("FIREWORKS_BASE_URL", raising=False)
    monkeypatch.delenv("ALLOWED_MODELS", raising=False)

    stub = types.ModuleType("local_model")
    stub.LocalLM = StubLM
    monkeypatch.setitem(sys.modules, "local_model", stub)

    import main
    main = importlib.reload(main)
    assert main.main() == 0

    stats = main.LAST_RUN_STATS
    # Histogram covers ALL tasks, not just those processed before bulk escalation.
    assert stats["categories"] == {"math": 1, "sentiment": 1, "factual": 1}
    assert sum(stats["paths"].values()) == 3
    # math verified + sentiment verified; factual answer "hm" is unverified but
    # non-empty, and with no Fireworks it stays local-unverified (not fallback).
    assert stats["paths"].get(main.LOCAL_VERIFIED) == 2
    assert stats["paths"].get(main.LOCAL_UNVERIFIED) == 1
    assert stats["fw_attempted"] == 0 and stats["fw_succeeded"] == 0
    assert stats["fw_tokens"] == 0

    results = json.loads(out.read_text())
    assert [r["task_id"] for r in results] == ["m1", "s1", "f1"]
    assert all(r["answer"].strip() for r in results)
