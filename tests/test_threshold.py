"""Threshold tuning + policy derivation (router/threshold.py, torch-free)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "router"))

from threshold import (MIN_UNVERIFIED_SAMPLES, derive_policy, pick_threshold,
                       simulate, wilson_lower_bound)  # noqa: E402


def _rec(category, correct, verified, score=0.0):
    return {"category": category, "correct": correct, "verified": verified,
            "score": score, "label": "local_ok" if correct else "escalate"}


def test_derive_policy_buckets():
    records = (
        # sentiment: a 90% point estimate is not enough for trust_local.
        [_rec("sentiment", True, False)] * 9
        + [_rec("sentiment", False, False)]
        # factual: 40/40 clears the 90% Wilson lower-bound requirement.
        + [_rec("factual", True, False)] * 40
        # logic: 5/10 correct -> always_escalate
        + [_rec("logic", True, True)] * 5 + [_rec("logic", False, False)] * 5
        # ner: 80% correct but unverified answers are coin flips -> neither
        + [_rec("ner", True, True)] * 6
        + [_rec("ner", True, False)] * 2 + [_rec("ner", False, False)] * 2
    )
    policy = derive_policy(records)
    assert "sentiment" not in policy["trust_local"]
    assert "factual" in policy["trust_local"]
    assert "logic" in policy["always_escalate"]
    assert "ner" not in policy["trust_local"]
    assert "ner" not in policy["always_escalate"]


def test_derive_policy_sparse_unverified_does_not_use_overall_accuracy():
    # 19/20 correct overall, but only 2 unverified samples: too little direct
    # evidence to bypass verification for the whole category.
    records = ([_rec("math", True, True)] * 18 + [_rec("math", True, False)]
               + [_rec("math", False, False)])
    policy = derive_policy(records)
    assert "math" not in policy["trust_local"]


def test_derive_policy_requires_minimum_unverified_sample_count():
    too_few = [_rec("factual", True, False)] * (MIN_UNVERIFIED_SAMPLES - 1)
    enough = [_rec("factual", True, False)] * 40
    assert "factual" not in derive_policy(too_few)["trust_local"]
    assert "factual" in derive_policy(enough)["trust_local"]
    assert wilson_lower_bound(10, 10) < 0.90
    assert wilson_lower_bound(40, 40) >= 0.90


def test_pick_threshold_escalates_verified_but_wrong():
    policy = {"always_escalate": [], "trust_local": []}
    # 8 easy tasks the router scores low; 2 poisoned tasks that pass
    # verification while wrong, scored high by the router.
    records = ([_rec("math", True, True, score=0.05)] * 8
               + [_rec("math", False, True, score=0.9)] * 2)
    thr, proj = pick_threshold(records, policy, max_expected_misses=1.0)
    assert thr <= 0.9  # must route the poisoned ones to Fireworks
    assert proj["expected_misses"] <= 1.0
    assert proj["expected_escalations"] >= 2 * 19 / 10 * 0.99


def test_pick_threshold_prefers_fewer_tokens_when_safe():
    policy = {"always_escalate": [], "trust_local": []}
    # Everything is verified-correct: no threshold can cause a miss, so the
    # cheapest plan (escalate nothing, threshold above every score) must win.
    records = [_rec("factual", True, True, score=s / 10) for s in range(10)]
    thr, proj = pick_threshold(records, policy, max_expected_misses=1.0)
    assert proj["expected_tokens"] == 0
    assert thr > 0.9


def test_simulate_accounting():
    policy = {"always_escalate": [], "trust_local": ["factual"]}
    records = [
        _rec("factual", True, False, score=0.1),   # trusted local, correct
        _rec("factual", False, False, score=0.1),  # trusted local, WRONG -> miss
        _rec("math", True, False, score=0.1),      # unverified, bounced -> escalates
        _rec("math", True, True, score=0.99),      # routed to Fireworks by score
    ]
    proj = simulate(records, threshold=0.5, policy=policy, eval_size=4,
                    esc_tokens=100, esc_accuracy=1.0)
    assert proj["expected_escalations"] == 2.0
    assert proj["expected_tokens"] == 200
    assert proj["expected_misses"] == 1.0
    assert proj["local_share"] == 0.5


def test_slow_failed_local_attempt_must_be_planned_upfront():
    policy = {"always_escalate": [], "trust_local": []}
    records = [
        {**_rec("summarization", False, False, score=0.2),
         "latency_s": 22.0},
        {**_rec("summarization", True, True, score=0.1),
         "latency_s": 2.0},
    ]
    threshold, projection = pick_threshold(
        records, policy, max_expected_misses=10.0)
    assert threshold <= 0.2
    assert projection["slow_failure_recall"] == 1.0
    assert projection["reactive_at_risk"] == 0.0


def test_latency_aware_projection_counts_at_risk_reactive_failure_as_miss():
    policy = {"always_escalate": [], "trust_local": []}
    records = [
        {**_rec("summarization", False, False, score=0.1),
         "latency_s": 22.0},
    ]
    projection = simulate(
        records, threshold=0.5, policy=policy, eval_size=1,
        esc_accuracy=1.0)
    assert projection["reactive_at_risk"] == 1.0
    assert projection["expected_fallback_misses"] == 1.0
    assert projection["expected_misses"] == 1.0
