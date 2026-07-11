"""Budget estimator + plan builder: the pure planning core of v2."""
import budget
from budget import (build_plan, estimate_local_cost, fits_deadline,
                    resolve_threshold)


def test_estimate_scales_with_speed_and_prompt():
    fast = estimate_local_cost("x" * 400, "factual", tok_s=20)
    slow = estimate_local_cost("x" * 400, "factual", tok_s=5)
    assert slow > fast
    short = estimate_local_cost("x" * 40, "factual", tok_s=8)
    long = estimate_local_cost("x" * 4000, "factual", tok_s=8)
    assert long > short


def test_estimate_uses_measured_tokens_over_cap():
    capped = estimate_local_cost("q", "codegen", tok_s=8)          # 400-token cap
    measured = estimate_local_cost("q", "codegen", tok_s=8,
                                   expected_tokens=80)             # p90 from labels
    assert measured < capped


def test_retry_headroom_applies_only_to_unmeasured_hard_caps():
    # Fallback hard caps need retry headroom; measured p90 token totals already
    # include any retry and therefore must not be multiplied again.
    code = estimate_local_cost("q", "codegen", tok_s=8)
    factual = estimate_local_cost("q", "factual", tok_s=8)
    assert code > factual
    measured_code = estimate_local_cost(
        "q", "codegen", tok_s=8, expected_tokens=100)
    measured_factual = estimate_local_cost(
        "q", "factual", tok_s=8, expected_tokens=100)
    assert measured_code == measured_factual


def test_estimate_uses_measured_p90_latency_as_a_floor():
    measured = {"completion_tokens": 10, "p90_latency_s": 17.5}
    assert estimate_local_cost(
        "short", "summarization", tok_s=20,
        expected_tokens=measured) == 17.5


def test_build_plan_splits_on_threshold_and_policy():
    prompts = ["a", "b", "c", "d"]
    cats = ["math", "factual", "logic", "ner"]
    scores = [0.9, 0.2, 0.4, 0.1]
    policy = {"always_escalate": ["logic"]}
    esc, local = build_plan(prompts, cats, scores, 0.5, policy)
    assert set(esc) == {0, 2}  # 0 by score, 2 by policy despite low score
    assert set(local) == {1, 3}


def test_build_plan_without_router_scores():
    prompts = ["a", "b"]
    cats = ["math", "factual"]
    esc, local = build_plan(prompts, cats, None, 0.5, {})
    assert esc == [] and set(local) == {0, 1}


def test_build_plan_local_queue_ascending_cost():
    prompts = ["short", "x" * 6000, "medium " * 10]
    cats = ["sentiment", "sentiment", "sentiment"]
    _, local = build_plan(prompts, cats, [0.0, 0.0, 0.0], 0.5, {})
    costs = [estimate_local_cost(prompts[i], cats[i], 8.0) for i in local]
    assert costs == sorted(costs)
    assert local[0] == 0 and local[-1] == 1


def test_fits_deadline_respects_reserve():
    assert fits_deadline(10, remaining_s=40, reserve_s=25)
    assert not fits_deadline(20, remaining_s=40, reserve_s=25)


def test_resolve_threshold_env_beats_config(monkeypatch):
    monkeypatch.setenv("ROUTER_THRESHOLD", "0.7")
    assert resolve_threshold({"threshold": 0.3}) == 0.7
    monkeypatch.delenv("ROUTER_THRESHOLD")
    assert resolve_threshold({"threshold": 0.3}) == 0.3
    assert resolve_threshold(None) == 0.5
    monkeypatch.setenv("ROUTER_THRESHOLD", "not-a-number")
    assert resolve_threshold(None) == 0.5
    for invalid in ("nan", "inf", "-0.01", "1.02"):
        monkeypatch.setenv("ROUTER_THRESHOLD", invalid)
        assert resolve_threshold({"threshold": 0.4}) == 0.4


def test_estimate_never_divides_by_zero():
    assert estimate_local_cost("", "math", tok_s=0) > 0
    assert budget.estimate_local_cost(None, "unknown-category", 8.0) > 0
