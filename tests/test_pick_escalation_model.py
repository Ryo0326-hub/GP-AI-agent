"""Accuracy-safe escalation-model evaluation and ranking."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "eval"))

import pick_escalation_model as picker  # noqa: E402


def _result(model, accuracy, tokens_per_correct, errors=0, style="raw"):
    return {
        "model": model,
        "mode": "default",
        "style": style,
        "n": 20,
        "correct": round(20 * accuracy),
        "accuracy": accuracy,
        "total_tokens": round(tokens_per_correct * max(1, 20 * accuracy)),
        "tokens_per_correct": tokens_per_correct,
        "call_errors": errors,
    }


def test_call_errors_remain_in_accuracy_denominator(monkeypatch):
    monkeypatch.setattr(picker, "HARD_TASKS", [
        ("ok-1", "expected", "contains"),
        ("network-error", "expected", "contains"),
        ("ok-2", "expected", "contains"),
    ])
    monkeypatch.setattr(picker, "grade", lambda *args: True)

    def fake_call(base, key, model, prompt, max_tokens, extra, system=None):
        if prompt == "network-error":
            return None, 0, "HTTP 500"
        return "expected", 10, None

    monkeypatch.setattr(picker, "call", fake_call)
    result = picker.eval_model("base", "key", "model", "default", {})

    assert result["n"] == 3
    assert result["correct"] == 2
    assert result["call_errors"] == 1
    assert result["accuracy"] == 0.667
    assert result["style"] == "raw"


def test_probe_error_is_recorded_as_failed_run(monkeypatch):
    monkeypatch.setattr(picker, "HARD_TASKS", [
        ("probe", "expected", "contains"),
        ("would-have-run", "expected", "contains"),
    ])
    monkeypatch.setattr(
        picker, "call",
        lambda *args, **kwargs: (None, 0, "HTTP 400"),
    )

    result = picker.eval_model("base", "key", "model", "unsupported", {})

    assert result["n"] == 2
    assert result["correct"] == 0
    assert result["accuracy"] == 0.0
    assert result["call_errors"] == 1
    assert result["tasks_not_run"] == 1
    assert result["probe_error"] == "HTTP 400"
    assert result["style"] == "raw"


def test_category_style_passes_exact_runtime_system_prompt(monkeypatch):
    prompt = "What is 2 + 2?"
    monkeypatch.setattr(picker, "HARD_TASKS", [
        (prompt, 4, "number"),
    ])
    seen = []

    def fake_call(base, key, model, user_prompt, max_tokens, extra, system=None):
        seen.append((user_prompt, system))
        return "Answer: 4", 25, None

    monkeypatch.setattr(picker, "call", fake_call)
    result = picker.eval_model(
        "base", "key", "model", "reasoning_effort=none",
        {"reasoning_effort": "none"}, prompt_style="category")

    assert result["correct"] == 1
    assert result["style"] == "category"
    assert seen == [(prompt, picker.SYSTEM["math"])]


def test_raw_style_has_no_system_prompt(monkeypatch):
    prompt = "Which country hosted the 2016 Summer Olympics, and in which city?"
    monkeypatch.setattr(picker, "HARD_TASKS", [
        (prompt, ["brazil", "rio"], "contains_all"),
    ])
    seen = []

    def fake_call(base, key, model, user_prompt, max_tokens, extra, system=None):
        seen.append(system)
        return "Brazil, in Rio de Janeiro.", 20, None

    monkeypatch.setattr(picker, "call", fake_call)
    result = picker.eval_model(
        "base", "key", "model", "default", {}, prompt_style="raw")

    assert result["correct"] == 1
    assert seen == [None]


def test_style_and_model_filters_are_exact():
    allowed = [
        "accounts/fireworks/models/deepseek-v4-flash",
        "accounts/fireworks/models/deepseek-v4-pro",
        "accounts/fireworks/models/another-model",
    ]
    assert picker.parse_styles("raw,category") == ["raw", "category"]
    assert picker.select_models(
        allowed, "deepseek-v4-flash,deepseek-v4-pro") == allowed[:2]


def test_call_converts_network_exception_to_failed_sample(monkeypatch):
    def fail(*args, **kwargs):
        raise picker.requests.ConnectionError("offline")

    monkeypatch.setattr(picker.requests, "post", fail)
    text, tokens, error = picker.call(
        "https://example.invalid", "secret", "model", "prompt", 10, {})

    assert text is None
    assert tokens == 0
    assert error == "network error: ConnectionError"


def test_ranking_prefers_accuracy_before_token_cost():
    cheap = _result("cheap", accuracy=0.90, tokens_per_correct=20,
                    style="raw")
    accurate = _result("accurate", accuracy=1.00, tokens_per_correct=200,
                       style="category")

    ranked = picker.rank_results([cheap, accurate], min_accuracy=0.90)

    assert ranked[0]["model"] == "accurate"
    assert ranked[0]["style"] == "category"


def test_error_free_safe_tier_beats_flaky_candidate():
    reliable = _result("reliable", accuracy=0.90, tokens_per_correct=100)
    flaky = _result("flaky", accuracy=0.95, tokens_per_correct=10, errors=1)

    ranked = picker.rank_results([flaky, reliable], min_accuracy=0.90)

    assert ranked[0]["model"] == "reliable"


def test_below_floor_fallback_still_uses_highest_accuracy():
    cheaper = _result("cheaper", accuracy=0.75, tokens_per_correct=10)
    better = _result("better", accuracy=0.85, tokens_per_correct=100)

    ranked = picker.rank_results([cheaper, better], min_accuracy=0.90)

    assert ranked[0]["model"] == "better"
