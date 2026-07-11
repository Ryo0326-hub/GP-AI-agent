import time
from concurrent.futures import ThreadPoolExecutor

import main


class ScriptedFW:
    available = True

    def __init__(self, responder):
        self.calls = 0
        self.systems = []
        self.responder = responder

    def complete(self, prompt, cap, timeout=25.0, system=None):
        self.calls += 1
        self.systems.append(system)
        return self.responder(prompt, self.calls)


def _run(fw, n=12, tmp_out=None):
    tasks = [{"task_id": f"t{i}", "prompt": f"Question {i}?"} for i in range(n)]
    answers, paths = {}, {}
    still = main.bulk_escalate(fw, tasks, list(range(n)), answers, paths,
                               time.monotonic() + 60)
    return still, answers, paths


def test_transient_probe_failures_continue_instead_of_mass_fallback(
        monkeypatch, tmp_path):
    monkeypatch.setattr(main, "OUTPUT_PATH", str(tmp_path / "r.json"))
    fw = ScriptedFW(lambda p, c: "")
    still, answers, paths = _run(fw)
    assert fw.calls == 12
    assert len(still) == 12 and not answers


def test_definitive_endpoint_failure_stops_after_probes(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "OUTPUT_PATH", str(tmp_path / "r.json"))
    fw = ScriptedFW(lambda p, c: "")
    fw.definitive_unavailable = True
    still, answers, paths = _run(fw)
    assert fw.calls == main.BULK_ABORT_AFTER
    assert len(still) == 12 and not answers


def test_healthy_endpoint_escalates_everything(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "OUTPUT_PATH", str(tmp_path / "r.json"))
    fw = ScriptedFW(lambda p, c: "answer")
    still, answers, paths = _run(fw)
    assert still == [] and len(answers) == 12
    assert all(v == main.ESCALATED for v in paths.values())


def test_partial_probe_success_continues(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "OUTPUT_PATH", str(tmp_path / "r.json"))
    # Exactly one of the first three probes succeeds -> keep going.
    fw = ScriptedFW(lambda p, c: "answer" if c != 1 else "")
    still, answers, paths = _run(fw)
    assert fw.calls == 12
    assert len(answers) == 11 and len(still) == 1


def test_bulk_escalation_passes_exact_category_system(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "OUTPUT_PATH", str(tmp_path / "r.json"))
    fw = ScriptedFW(lambda p, c: "answer")
    tasks = [{"task_id": "m1", "prompt": "What is 2+2? Calculate the total."}]
    answers, paths = {}, {}

    with ThreadPoolExecutor(max_workers=1) as ex:
        successes = main._escalate_batch(
            ex, fw, tasks, [0], answers, paths, time.monotonic() + 60)

    assert successes == 1
    assert fw.systems == [main.SYSTEM["math"]]
