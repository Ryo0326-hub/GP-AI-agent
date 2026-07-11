import importlib


def _load_main(monkeypatch, **env):
    defaults = {"TOTAL_DEADLINE_S": "540", "PER_TASK_CAP_S": "25",
                "PER_TASK_MIN_S": "20"}
    defaults.update({k: str(v) for k, v in env.items()})
    for k, v in defaults.items():
        monkeypatch.setenv(k, v)
    import main
    return importlib.reload(main)


def test_80_tasks_default_deadline_triggers_bulk_not_truncation(monkeypatch):
    main = _load_main(monkeypatch)
    mode, budget = main.plan_budget(remaining=540, todo=80, fw_available=True)
    assert mode == "bulk" and budget is None


def test_no_fireworks_degrades_instead_of_skipping(monkeypatch):
    main = _load_main(monkeypatch)
    mode, budget = main.plan_budget(remaining=540, todo=80, fw_available=False)
    assert mode == "local"
    assert 4.0 <= budget < main.PER_TASK_MIN_S  # degraded, but never zero


def test_19_tasks_get_full_budgets_never_below_floor(monkeypatch):
    main = _load_main(monkeypatch)
    mode, budget = main.plan_budget(remaining=540, todo=19, fw_available=True)
    assert mode == "local"
    assert main.PER_TASK_MIN_S <= budget <= main.PER_TASK_CAP_S


def test_floor_times_todo_exceeding_remaining_is_the_trigger(monkeypatch):
    main = _load_main(monkeypatch)
    # 10 tasks x 20s floor + 15s margin = 215s needed
    assert main.plan_budget(remaining=230, todo=10, fw_available=True)[0] == "local"
    assert main.plan_budget(remaining=210, todo=10, fw_available=True)[0] == "bulk"


def test_accuracy_mode_env_overrides(monkeypatch):
    main = _load_main(monkeypatch, TOTAL_DEADLINE_S="7200", PER_TASK_CAP_S="300")
    mode, budget = main.plan_budget(remaining=7200, todo=80, fw_available=True)
    assert mode == "local"
    assert budget == 90  # 7200/80, within [20, 300]
