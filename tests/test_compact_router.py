"""Compact router hashing, scoring, loading, and benchmark exclusion."""
import json
import math
import os
import sys

import pytest

import router as runtime_router


ROUTER_TOOLS = os.path.join(os.path.dirname(__file__), "..", "router")
if ROUTER_TOOLS not in sys.path:
    sys.path.insert(0, ROUTER_TOOLS)
from train_compact_router import (  # noqa: E402
    derive_training_runtime_config,
    exclude_prompt_matches,
    exclude_template_groups,
    load_label_records,
    load_excluded_tasks,
    load_excluded_prompts,
    main as train_compact_main,
    validate_required_task_coverage,
)


def _profile(**updates):
    profile = {
        "schemaVersion": 1,
        "revision": "unit-test",
        "modelType": "hashed-logistic-v1",
        "dimension": 2048,
        "bias": 0.0,
        "threshold": 0.5,
        "trainedExamples": 2,
        "weights": {},
        "metrics": {},
    }
    profile.update(updates)
    return profile


def test_fnv1a_matches_browser_textencoder_math_imul_fixtures():
    assert runtime_router.fnv1a_32("") == 2166136261
    assert runtime_router.fnv1a_32("hello") == 1335831723
    assert runtime_router.fnv1a_32("category:math") == 4216891947
    # Non-BMP fixture guards the JS /u + Array.from parity contract.
    assert runtime_router.fnv1a_32("u:🧪") == 784774262
    assert runtime_router.fnv1a_32("u:\ud83e\uddea") == 784774262


def test_feature_hashes_match_browser_fixture_including_collision_count():
    assert runtime_router.compact_feature_counts(
        "What is 2 + 2?", "math", 2048,
    ) == {
        113: 1,
        500: 1,
        555: 1,
        584: 1,
        632: 1,
        722: 1,
        835: 1,
        951: 1,
        1032: 1,
        1294: 1,
        1391: 1,
        1428: 1,
        1523: 1,
        1860: 2,
    }


def test_compact_score_uses_log_count_transform_and_sigmoid():
    profile = _profile(
        bias=-1.0,
        weights={"1860": 2.0},
    )
    score = runtime_router.score_compact_prompt(
        "What is 2 + 2?", profile, category="math",
    )
    logit = -1.0 + 2.0 * (1.0 + math.log(2.0))
    assert score == pytest.approx(1.0 / (1.0 + math.exp(-logit)))


def test_profile_loading_and_config_use_compact_calibrated_threshold(
        tmp_path, monkeypatch):
    monkeypatch.delenv("COMPACT_ROUTER_PROFILE", raising=False)
    (tmp_path / "router_config.json").write_text(json.dumps({
        "threshold": 0.2,
        "category_policy": {"always_escalate": ["logic"]},
    }))
    (tmp_path / "compact_router.json").write_text(json.dumps(
        _profile(threshold=0.73)))

    loaded = runtime_router.load_compact_profile(str(tmp_path))
    assert loaded["threshold"] == 0.73
    assert loaded["weights"] == {}
    config = runtime_router.load_config(str(tmp_path))
    assert config["threshold"] == 0.73
    assert config["category_policy"] == {"always_escalate": ["logic"]}
    assert config["router_model_type"] == "hashed-logistic-v1"


def test_compact_only_config_exposes_train_policy_and_expected_tokens(
        tmp_path, monkeypatch):
    monkeypatch.delenv("COMPACT_ROUTER_PROFILE", raising=False)
    profile = _profile(
        threshold=0.61,
        categoryPolicy={
            "always_escalate": ["math"],
            "trust_local": ["factual"],
        },
        expectedCompletionTokens={"math": 144, "factual": 32},
        expectedLocalLatencySeconds={"math": 18.0, "factual": 3.0},
    )
    (tmp_path / "compact_router.json").write_text(json.dumps(profile))

    config = runtime_router.load_config(str(tmp_path))
    assert config == {
        "threshold": 0.61,
        "router_model_type": "hashed-logistic-v1",
        "router_revision": "unit-test",
        "category_policy": {
            "always_escalate": ["math"],
            "trust_local": ["factual"],
        },
        "category_policy_source": "compact_profile_train",
        "expected_completion_tokens": {
            "math": {"completion_tokens": 144.0, "p90_latency_s": 18.0},
            "factual": {"completion_tokens": 32.0, "p90_latency_s": 3.0},
        },
        "expected_completion_tokens_source": "compact_profile_train",
    }


def test_runtime_policy_and_token_p90_are_derived_from_train_records_only():
    train = [
        {
            "category": "factual",
            "correct": True,
            "verified": False,
            "completion_tokens": token_count,
        }
        for token_count in range(1, 11)
    ] + [
        {
            "category": "math",
            "correct": False,
            "verified": False,
            "completion_tokens": token_count,
        }
        for token_count in (80, 120, 160)
    ]
    policy, expected, stats = derive_training_runtime_config(train)

    assert policy == {
        "always_escalate": ["math"],
        "trust_local": [],
    }
    assert expected == {"factual": 10, "math": 160}
    assert stats["factual"]["n"] == 10
    assert stats["math"]["accuracy"] == 0.0


def test_score_and_free_prefers_compact_and_invalid_profile_falls_back(
        tmp_path, monkeypatch):
    monkeypatch.delenv("COMPACT_ROUTER_PROFILE", raising=False)
    compact = _profile(bias=2.0)
    (tmp_path / "compact_router.json").write_text(json.dumps(compact))
    monkeypatch.setattr(
        runtime_router, "_score",
        lambda *args: pytest.fail("DistilBERT should not load"),
    )
    scores = runtime_router.score_and_free(["hello"], str(tmp_path))
    assert scores == [pytest.approx(1.0 / (1.0 + math.exp(-2.0)))]

    (tmp_path / "compact_router.json").write_text("{broken")
    (tmp_path / "config.json").write_text("{}")
    monkeypatch.setattr(runtime_router, "_score", lambda *args: [0.42])
    assert runtime_router.score_and_free(
        ["hello"], str(tmp_path)) == [0.42]


def test_supplied_tasks_json_prompts_are_excluded_before_training(tmp_path):
    tasks_path = tmp_path / "tasks.json"
    tasks_path.write_text(json.dumps([
        {"task_id": "judge-1", "prompt": "  WHAT is 2 + 2?\n"},
    ]))
    records = [
        {"task_id": "leak", "prompt": "What is 2 + 2?"},
        {"task_id": "keep", "prompt": "What is 3 + 3?"},
    ]
    excluded_prompts = load_excluded_prompts([str(tasks_path)])
    kept, excluded = exclude_prompt_matches(records, excluded_prompts)
    assert [record["task_id"] for record in kept] == ["keep"]
    assert [record["task_id"] for record in excluded] == ["leak"]


def test_public_template_match_excludes_the_entire_generated_group(tmp_path):
    tasks_path = tmp_path / "public.json"
    tasks_path.write_text(json.dumps([{
        "task_id": "codegen_999",
        "prompt": ("Write a Python function that returns the second-largest "
                   "number in a list, handling duplicates correctly."),
    }]))
    records = [
        {
            "task_id": "codegen_001", "dataset_category": "codegen",
            "category": "codegen",
            "prompt": ("Write a Python function that returns the second-largest "
                       "number in a list, handling duplicates correctly."),
        },
        {
            "task_id": "codegen_002", "dataset_category": "codegen",
            "category": "codegen",
            "prompt": ("Implement a Python function that returns the second-largest "
                       "number in a list, handling duplicates correctly."),
        },
        {
            "task_id": "codegen_003", "dataset_category": "codegen",
            "category": "codegen",
            "prompt": "Write a Python function that computes factorial.",
        },
    ]
    public = load_excluded_tasks([str(tasks_path)])
    kept, excluded, diagnostics = exclude_template_groups(records, public, 0.82)
    assert [record["task_id"] for record in kept] == ["codegen_003"]
    assert {record["task_id"] for record in excluded} == {
        "codegen_001", "codegen_002"}
    assert diagnostics["excludedGroupCount"] == 1
    assert diagnostics["matchedGroups"][0]["taskIds"] == [
        "codegen_001", "codegen_002"]


def test_required_task_coverage_accepts_complete_normalized_manifest(tmp_path):
    tasks_path = tmp_path / "tasks.json"
    tasks_path.write_text(json.dumps({"tasks": [
        {"task_id": "a", "prompt": "Caf\u00e9   question"},
        {"task_id": "b", "prompt": "Second prompt"},
    ]}))
    records = [
        {"task_id": "a", "prompt": "  CAFE\u0301\nQUESTION  "},
        {"task_id": "b", "prompt": "second prompt"},
    ]
    assert validate_required_task_coverage(records, tasks_path) == 2


def test_required_task_coverage_reports_each_mismatch_bucket(tmp_path):
    tasks_path = tmp_path / "tasks.json"
    tasks_path.write_text(json.dumps([
        {"task_id": "a", "prompt": "prompt a"},
        {"task_id": "b", "prompt": "prompt b"},
    ]))
    complete = [
        {"task_id": "a", "prompt": "prompt a"},
        {"task_id": "b", "prompt": "prompt b"},
    ]

    with pytest.raises(ValueError, match="missing=.*b"):
        validate_required_task_coverage(complete[:1], tasks_path)
    with pytest.raises(ValueError, match="extras=.*c"):
        validate_required_task_coverage(
            complete + [{"task_id": "c", "prompt": "prompt c"}], tasks_path)
    with pytest.raises(ValueError, match="duplicates=.*a"):
        validate_required_task_coverage(complete + [complete[0]], tasks_path)
    stale = [dict(record) for record in complete]
    stale[1]["prompt"] = "stale prompt"
    with pytest.raises(ValueError, match="prompt_mismatches=.*b"):
        validate_required_task_coverage(stale, tasks_path)


def test_required_task_coverage_rejects_bad_manifest_and_duplicate_ids(tmp_path):
    tasks_path = tmp_path / "tasks.json"
    tasks_path.write_text(json.dumps({"not_tasks": []}))
    with pytest.raises(ValueError, match="must be an array"):
        validate_required_task_coverage([], tasks_path)

    tasks_path.write_text(json.dumps([
        {"task_id": "same", "prompt": "one"},
        {"task_id": "same", "prompt": "two"},
    ]))
    with pytest.raises(ValueError, match="duplicate required task_id"):
        validate_required_task_coverage([], tasks_path)


def test_require_tasks_cli_fails_before_writing_partial_artifact(tmp_path):
    tasks_path = tmp_path / "tasks.json"
    tasks_path.write_text(json.dumps([
        {"task_id": "a", "prompt": "prompt a"},
        {"task_id": "b", "prompt": "prompt b"},
    ]))
    labels_path = tmp_path / "labels.jsonl"
    provenance = {
        "schema_version": 1,
        "gguf_sha256": "0" * 64,
        "solver_bundle_sha256": "0" * 64,
        "labeler_sha256": "0" * 64,
        "task_manifest_sha256": "0" * 64,
        "expected_answers_sha256": "0" * 64,
        "budget_seconds": 18.0,
        "threads": 2,
    }
    labels_path.write_text(json.dumps({
        "task_id": "a", "prompt": "prompt a", "label": "local_ok",
        "dataset_category": "factual", "category": "factual",
        "correct": True, "verified": True, "latency_s": 1.0,
        "completion_tokens": 5, "model": "unit",
        "provenance": provenance,
    }) + "\n")
    out_path = tmp_path / "compact_router.json"

    with pytest.raises(ValueError, match="required task coverage mismatch"):
        train_compact_main([
            "--labels", str(labels_path),
            "--require-tasks", str(tasks_path),
            "--out", str(out_path),
            "--demo-out", "",
        ])
    assert not out_path.exists()


def test_strict_label_schema_rejects_inconsistent_or_stale_records(tmp_path):
    base = {
        "task_id": "a", "prompt": "Explain photosynthesis.",
        "dataset_category": "factual", "category": "factual",
        "label": "local_ok", "correct": True, "verified": True,
        "latency_s": 1.0, "completion_tokens": 5, "model": "unit",
        "provenance": {
            "schema_version": 1,
            "gguf_sha256": "0" * 64,
            "solver_bundle_sha256": "1" * 64,
            "labeler_sha256": "2" * 64,
            "task_manifest_sha256": "3" * 64,
            "expected_answers_sha256": "4" * 64,
            "budget_seconds": 18.0,
            "threads": 2,
        },
    }
    path = tmp_path / "labels.jsonl"
    for field, value, message in (
            ("correct", 1, "correct must be boolean"),
            ("label", "escalate", "label/correct mismatch"),
            ("category", "math", "does not match runtime classifier"),
            ("latency_s", float("nan"), "latency_s must be finite"),
            ("completion_tokens", -1, "completion_tokens must be")):
        record = dict(base)
        record[field] = value
        path.write_text(json.dumps(record) + "\n")
        with pytest.raises(ValueError, match=message):
            load_label_records([path])
