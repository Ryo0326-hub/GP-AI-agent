"""Per-task Fireworks deadlines include queue and local-solve time."""
import main


class StubFW:
    def __init__(self):
        self.timeouts = []
        self.systems = []

    def complete(self, prompt, cap, timeout=25.0, system=None):
        self.timeouts.append(timeout)
        self.systems.append(system)
        return "answer"


def test_escalation_timeout_is_always_strictly_below_30_seconds(monkeypatch):
    fw = StubFW()
    monkeypatch.setattr(main, "PER_TASK_CAP_S", 300.0)
    monkeypatch.setattr(main, "ESCALATION_TIMEOUT_S", 90.0)
    monkeypatch.setattr(main.time, "monotonic", lambda: 100.0)

    answer = main._complete_with_task_deadline(
        fw, "prompt", 100, task_started_at=100.0, deadline=1000.0)

    assert answer == "answer"
    assert len(fw.timeouts) == 1
    assert fw.timeouts[0] == main.MAX_REQUEST_S
    assert fw.timeouts[0] < 30.0


def test_queue_and_local_time_reduce_remote_allowance(monkeypatch):
    fw = StubFW()
    monkeypatch.setattr(main, "PER_TASK_CAP_S", 25.0)
    monkeypatch.setattr(main, "ESCALATION_TIMEOUT_S", 25.0)
    # The task began at 100; ten seconds were already spent locally/in queue.
    monkeypatch.setattr(main.time, "monotonic", lambda: 110.0)

    main._complete_with_task_deadline(
        fw, "prompt", 100, task_started_at=100.0, deadline=1000.0)

    assert fw.timeouts == [15.0]


def test_expired_task_is_not_sent_to_fireworks(monkeypatch):
    fw = StubFW()
    monkeypatch.setattr(main, "PER_TASK_CAP_S", 25.0)
    monkeypatch.setattr(main, "ESCALATION_TIMEOUT_S", 25.0)
    monkeypatch.setattr(main.time, "monotonic", lambda: 126.0)

    answer = main._complete_with_task_deadline(
        fw, "prompt", 100, task_started_at=100.0, deadline=1000.0)

    assert answer == ""
    assert fw.timeouts == []


def test_deadline_wrapper_preserves_exact_system_prompt(monkeypatch):
    fw = StubFW()
    monkeypatch.setattr(main.time, "monotonic", lambda: 100.0)

    main._complete_with_task_deadline(
        fw, "prompt", 100, task_started_at=100.0, deadline=1000.0,
        system=main.SYSTEM["math"])

    assert fw.systems == [main.SYSTEM["math"]]
