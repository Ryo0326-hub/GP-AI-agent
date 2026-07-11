"""Submission-time compact-router artifact safeguards."""
import json
import os
import sys

import pytest


EVAL_TOOLS = os.path.join(os.path.dirname(__file__), "..", "eval")
if EVAL_TOOLS not in sys.path:
    sys.path.insert(0, EVAL_TOOLS)

from check_router_artifacts import (  # noqa: E402
    check_router_artifacts,
    main as artifact_check_main,
)
from classifier import CATEGORIES  # noqa: E402
from router import fnv1a_32, score_compact_prompt  # noqa: E402
from threshold import policy_safety_metrics, simulate  # noqa: E402
from train_compact_router import (  # noqa: E402
    _profile_revision,
    build_training_provenance,
    classification_metrics,
    file_sha256,
    profile_semantic_sha256,
    solver_bundle_sha256,
)


def _resign(profile):
    profile.get("provenance", {}).pop("profileSemanticSha256", None)
    profile["revision"] = _profile_revision(
        profile, profile["split"]["task_ids"]["train"])
    profile["provenance"]["profileSemanticSha256"] = \
        profile_semantic_sha256(profile)


def _fixture(tmp_path, mutate_profile=None):
    tasks_path = tmp_path / "tasks.json"
    labels_path = tmp_path / "labels.jsonl"
    core_path = tmp_path / "core.json"
    demo_path = tmp_path / "demo.json"

    tasks = []
    for index in range(40):
        tasks.append({
            "task_id": f"factual_{index:03d}",
            "prompt": f"Explain photosynthesis variant alpha {index}.",
        })
        tasks.append({
            "task_id": f"math_{index:03d}",
            "prompt": f"Calculate 2 plus {index}.",
        })
    tasks_path.write_text(json.dumps(tasks))
    run_provenance = {
        "schema_version": 1,
        "gguf_sha256": "a" * 64,
        "solver_bundle_sha256": solver_bundle_sha256(),
        "labeler_sha256": file_sha256(
            os.path.join(os.path.dirname(__file__), "..", "eval", "label_local.py")),
        "task_manifest_sha256": file_sha256(tasks_path),
        "expected_answers_sha256": "b" * 64,
        "budget_seconds": 18.0,
        "threads": 2,
    }
    labels = []
    for task in tasks:
        is_math = task["task_id"].startswith("math_")
        labels.append({
            **task,
            "dataset_category": "math" if is_math else "factual",
            "category": "math" if is_math else "factual",
            "label": "escalate" if is_math else "local_ok",
            "correct": not is_math,
            "verified": not is_math,
            "latency_s": 20.0 if is_math else 1.0,
            "completion_tokens": 40 if is_math else 8,
            "model": "unit-1.5b",
            "provenance": run_provenance,
        })
    labels_path.write_text("".join(
        json.dumps(record) + "\n" for record in labels))

    dimension = 65536
    math_index = fnv1a_32("category:math") % dimension
    profile = {
        "schemaVersion": 1,
        "revision": "pending",
        "modelType": "hashed-logistic-v1",
        "dimension": dimension,
        "bias": -5.0,
        "threshold": 0.5,
        "trainedExamples": 20,
        "totalExamples": len(labels),
        "excludedExamples": 0,
        "weights": {str(math_index): 10.0},
        "categoryPolicy": {"always_escalate": ["math"], "trust_local": []},
        "expectedCompletionTokens": {category: 10 for category in CATEGORIES},
        "expectedLocalLatencySeconds": {category: 1.0 for category in CATEGORIES},
        "categoryStats": {category: {"n": 1} for category in CATEGORIES},
        "exclusion": {
            "strategy": "grouped_public_template_v1",
            "similarityThreshold": 0.82,
            "excludedRecordCount": 0,
            "matchedGroups": [],
        },
        "training": {
            "escalationTokens": 168.0,
            "escalationAccuracy": 0.95,
            "reactiveDeadlineSeconds": 25.0,
            "minimumRemoteWindowSeconds": 8.0,
        },
    }
    factual_ids = [record["task_id"] for record in labels
                   if record["category"] == "factual"]
    math_ids = [record["task_id"] for record in labels
                if record["category"] == "math"]
    train_ids = factual_ids[:10] + math_ids[:10]
    calibration_ids = factual_ids[10:20] + math_ids[10:20]
    test_ids = factual_ids[20:] + math_ids[20:]
    profile["split"] = {
        "task_ids": {
            "train": sorted(train_ids),
            "calibration": sorted(calibration_ids),
            "test": sorted(test_ids),
        },
    }
    by_id = {record["task_id"]: record for record in labels}
    test_records = []
    for task_id in test_ids:
        record = dict(by_id[task_id])
        record["score"] = score_compact_prompt(record["prompt"], profile)
        test_records.append(record)
    test_metrics = classification_metrics(test_records, profile["threshold"])
    assert test_metrics["accuracy"] == 1.0
    profile["metrics"] = {"test": test_metrics}
    profile["testProjection19"] = simulate(
        test_records, profile["threshold"], profile["categoryPolicy"],
        esc_tokens=168.0, esc_accuracy=0.95)
    profile["safetyEvaluation"] = {
        "test": policy_safety_metrics(
            test_records, profile["threshold"], profile["categoryPolicy"]),
    }
    profile["provenance"] = build_training_provenance(
        labels, [labels_path], tasks_path, [])
    _resign(profile)

    if mutate_profile:
        mutate_profile(profile)
    payload = json.dumps(profile, indent=2, sort_keys=True) + "\n"
    core_path.write_text(payload)
    demo_path.write_text(payload)
    return core_path, demo_path, labels_path, tasks_path


def test_identical_complete_trained_artifacts_pass(tmp_path):
    paths = _fixture(tmp_path)
    report = check_router_artifacts(*paths)
    assert report["labels"] == 80
    assert report["trained"] == 20
    assert report["test"] == 40
    assert report["balanced_accuracy"] == 1.0
    assert len(report["sha256"]) == 64


def test_artifact_check_rejects_core_demo_byte_mismatch(tmp_path):
    core, demo, labels, tasks = _fixture(tmp_path)
    demo.write_text(demo.read_text() + "\n")
    with pytest.raises(ValueError, match="core/demo router profiles differ"):
        check_router_artifacts(core, demo, labels, tasks)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda p: p.update(revision="training-pending"),
         "revision is missing or pending"),
        (lambda p: p.update(weights={}), "weights must be non-empty"),
        (lambda p: p.update(trainedExamples=0),
         "trainedExamples must be a positive"),
        (lambda p: p.update(categoryStats={}),
         "categoryStats missing categories"),
        (lambda p: p["metrics"]["test"].update(count=0),
         "metrics.test.count"),
    ],
)
def test_artifact_check_rejects_placeholder_or_incomplete_profile(
        tmp_path, mutation, message):
    paths = _fixture(tmp_path, mutation)
    with pytest.raises(ValueError, match=message):
        check_router_artifacts(*paths)


def test_artifact_check_rejects_label_manifest_or_profile_count_mismatch(tmp_path):
    core, demo, labels, tasks = _fixture(tmp_path)
    labels.write_text("\n".join(labels.read_text().splitlines()[:-1]) + "\n")
    with pytest.raises(ValueError, match="required task coverage mismatch"):
        check_router_artifacts(core, demo, labels, tasks)

    paths = _fixture(
        tmp_path, lambda profile: profile.update(totalExamples=79))
    with pytest.raises(ValueError, match="router profile label coverage mismatch"):
        check_router_artifacts(*paths)


def test_artifact_check_rejects_tampered_labels_metrics_and_provenance(tmp_path):
    core, demo, labels, tasks = _fixture(tmp_path)
    records = [json.loads(line) for line in labels.read_text().splitlines()]
    records[0]["latency_s"] = 2.0
    labels.write_text("".join(json.dumps(record) + "\n" for record in records))
    with pytest.raises(ValueError, match="labelRecordsSha256 mismatch"):
        check_router_artifacts(core, demo, labels, tasks)

    def tamper_metric(profile):
        profile["metrics"]["test"].update(accuracy=0.99)
        _resign(profile)

    paths = _fixture(tmp_path, tamper_metric)
    with pytest.raises(ValueError, match="does not match recomputed"):
        check_router_artifacts(*paths)

    paths = _fixture(
        tmp_path,
        lambda profile: profile["provenance"].update(
            solverBundleSha256="0" * 64),
    )
    with pytest.raises(ValueError, match="solver provenance mismatch"):
        check_router_artifacts(*paths)


def test_artifact_check_rejects_overlap_and_unsafe_policy(tmp_path):
    def overlap(profile):
        duplicate = profile["split"]["task_ids"]["train"][0]
        profile["split"]["task_ids"]["test"].append(duplicate)
        _resign(profile)

    with pytest.raises(ValueError, match="split task IDs overlap"):
        check_router_artifacts(*_fixture(tmp_path, overlap))

    def unsafe(profile):
        profile["categoryPolicy"] = {
            "always_escalate": [], "trust_local": ["math"]}
        profile["threshold"] = 1.01
        _resign(profile)

    with pytest.raises(ValueError, match=(
            "does not match recomputed|balanced accuracy|escalate recall|"
            "expected misses|unsafe-local")):
        check_router_artifacts(*_fixture(tmp_path, unsafe))


def test_artifact_check_enforces_predeclared_raw_recall_gate(tmp_path):
    def low_recall(profile):
        profile["metrics"]["test"]["escalateRecall"] = 0.79

    with pytest.raises(ValueError, match="escalate recall below safety gate"):
        check_router_artifacts(*_fixture(tmp_path, low_recall))


def test_artifact_check_cli_returns_nonzero_for_empty_artifact(tmp_path, capsys):
    core, demo, labels, tasks = _fixture(tmp_path)
    core.write_bytes(b"")
    assert artifact_check_main([
        "--core", str(core), "--demo", str(demo),
        "--labels", str(labels), "--require-tasks", str(tasks),
    ]) == 1
    assert "router artifact check failed" in capsys.readouterr().err
