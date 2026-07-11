import json

import main


def test_late_record_cannot_replace_final_complete_output(monkeypatch, tmp_path):
    output = tmp_path / "results.json"
    monkeypatch.setattr(main, "OUTPUT_PATH", str(output))
    monkeypatch.setattr(main, "_FINALIZED", False)
    tasks = [
        {"task_id": "a", "prompt": "A"},
        {"task_id": "b", "prompt": "B"},
    ]
    answers = {0: "first", 1: main.FALLBACK_ANSWER}
    paths = {0: main.LOCAL_VERIFIED, 1: main.FALLBACK}
    final = [
        {"task_id": "a", "answer": answers[0]},
        {"task_id": "b", "answer": answers[1]},
    ]

    with main._WRITE_LOCK:
        main._FINALIZED = True
        main.atomic_write_json(main.OUTPUT_PATH, final)

    assert main.record(
        1, "late provider answer", main.ESCALATED,
        tasks, answers, paths, overwrite=True,
    ) is False
    assert json.loads(output.read_text()) == final
