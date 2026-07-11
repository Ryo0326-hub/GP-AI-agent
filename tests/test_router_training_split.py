"""Leak-resistant router splits and calibration (torch/model-download free)."""
import os
import sys


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "router"))

from train_router import (  # noqa: E402
    calibrate_policy_and_threshold,
    canonical_prompt_template,
    group_prompt_templates,
    grouped_train_calibration_test_split,
    metrics_from_scores,
    summarize_training_categories,
)


def _rec(task_id, prompt, category="math", label="local_ok", **extra):
    record = {
        "task_id": task_id,
        "dataset_category": category,
        "category": category,
        "prompt": prompt,
        "label": label,
        "correct": label == "local_ok",
        "verified": label == "local_ok",
        "completion_tokens": 10,
    }
    record.update(extra)
    return record


def _group_for(groups, task_id):
    for group in groups:
        if any(record["task_id"] == task_id for record in group):
            return {record["task_id"] for record in group}
    raise AssertionError(f"task {task_id} was not grouped")


def test_canonical_templates_group_generated_siblings():
    records = [
        _rec("math-a", "A warehouse holds 240 boxes. 10% are shipped on Monday "
             "and 25 more on Tuesday. How many boxes remain?"),
        _rec("math-b", "A warehouse holds 720 boxes. 30% are shipped on Monday "
             "and 80 more on Tuesday. How many boxes remain?"),
        _rec("code-a", "Write a Python function that returns the second-largest "
             "number in a list, handling duplicates correctly.", category="codegen"),
        _rec("code-b", "Implement a Python function that returns the second-largest "
             "number in a list, handling duplicates correctly.", category="codegen"),
        _rec("sum-a", "Summarize the following paragraph in exactly one sentence: "
             "Honeybees pollinate crops and are threatened by habitat loss.",
             category="summarization"),
        _rec("sum-b", "Summarize the following paragraph in no more than 25 words: "
             "Honeybees pollinate crops and are threatened by habitat loss.",
             category="summarization"),
    ]
    groups = group_prompt_templates(records)
    assert _group_for(groups, "math-a") == {"math-a", "math-b"}
    assert _group_for(groups, "code-a") == {"code-a", "code-b"}
    assert _group_for(groups, "sum-a") == {"sum-a", "sum-b"}


def test_explicit_template_id_wins_over_dissimilar_prompt_text():
    records = [
        _rec("a", "Completely different wording", template_id="family-7"),
        _rec("b", "Nothing lexically similar", template_id="family-7"),
    ]
    assert canonical_prompt_template(records[0]) == \
        canonical_prompt_template(records[1])
    groups = group_prompt_templates(records)
    assert len(groups) == 1
    assert {r["task_id"] for r in groups[0]} == {"a", "b"}


def test_grouped_split_has_no_template_leakage_and_is_deterministic():
    records = []
    for i in range(30):
        category = "math" if i % 2 else "logic"
        label = "escalate" if i % 5 == 0 else "local_ok"
        for variant in range(2):
            records.append(_rec(
                f"task-{i}-{variant}", f"prompt {i}, variant {variant}",
                category=category, label=label, template_id=f"template-{i}",
            ))

    train, calibration, test, diagnostics = \
        grouped_train_calibration_test_split(
            records, seed=19, return_diagnostics=True,
        )
    split_by_task = {
        record["task_id"]: split
        for split, split_records in (
            ("train", train), ("calibration", calibration), ("test", test)
        )
        for record in split_records
    }
    for i in range(30):
        assert len({split_by_task[f"task-{i}-{variant}"]
                    for variant in range(2)}) == 1
    assert train and calibration and test
    digests = diagnostics["group_digests"]
    assert set(digests["train"]).isdisjoint(digests["calibration"])
    assert set(digests["train"]).isdisjoint(digests["test"])
    assert set(digests["calibration"]).isdisjoint(digests["test"])

    reversed_parts = grouped_train_calibration_test_split(
        list(reversed(records)), seed=19,
    )
    for expected, actual in zip((train, calibration, test), reversed_parts):
        assert {r["task_id"] for r in expected} == \
            {r["task_id"] for r in actual}


def test_policy_is_fit_from_train_and_threshold_from_calibration_only():
    train = [
        _rec(f"train-{i}", f"review {i}", category="sentiment",
             verified=False)
        for i in range(10)
    ] + [
        _rec("train-wrong", "bad review", category="sentiment",
             label="escalate", verified=True)
    ]
    calibration = [
        _rec("cal-ok", "easy", category="sentiment", score=0.1),
        _rec("cal-bad", "hard", category="sentiment", label="escalate",
             score=0.9, verified=True),
    ]

    policy, threshold, projection = calibrate_policy_and_threshold(
        train, calibration, max_expected_misses=10,
    )
    # Calibration is 50% wrong, but cannot alter this train-derived category
    # policy. The scored calibration examples alone determine the threshold.
    # Ten perfect examples are still too sparse for a 90% Wilson lower bound.
    assert policy["trust_local"] == []
    assert threshold in {0.0, 0.1, 0.5, 0.9, 1.01}
    assert projection["threshold"] == threshold


def test_metrics_and_category_summary_are_pure_and_threshold_specific():
    records = [
        _rec("ok", "easy", score=0.4, completion_tokens=10),
        _rec("esc", "hard", label="escalate", score=0.6,
             completion_tokens=100),
    ]
    assert metrics_from_scores(records, threshold=0.5)["accuracy"] == 1.0
    assert metrics_from_scores(records, threshold=0.7)["accuracy"] == 0.5
    summary = summarize_training_categories(records)
    assert summary["math"]["n"] == 2
    assert summary["math"]["p90_completion_tokens"] == 100
    assert summary["math"]["p90_latency_s"] == 0.0
