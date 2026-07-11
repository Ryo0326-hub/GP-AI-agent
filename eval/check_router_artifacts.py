#!/usr/bin/env python3
"""Fail-closed validation for submission router artifacts.

The Docker runtime and browser demo must ship the exact same trained profile,
and that profile must be derived from one label for every generated task.
This checker is intentionally network-free and uses only the training/runtime
validation code already in the repository.
"""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ROUTER_TOOLS = ROOT / "router"
APP_DIR = ROOT / "app"
for directory in (ROUTER_TOOLS, APP_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from classifier import CATEGORIES  # noqa: E402
from router import score_compact_prompt, validate_compact_profile  # noqa: E402
from train_compact_router import (  # noqa: E402
    _profile_revision,
    canonical_records_sha256,
    classification_metrics,
    file_sha256,
    load_label_records,
    profile_semantic_sha256,
    solver_bundle_sha256,
    validate_required_task_coverage,
)
from threshold import policy_safety_metrics, simulate  # noqa: E402


# Predeclared submission gates.  They are intentionally fixed in code before
# the final held-out split is inspected; a failure means collect new blind data,
# not tune against this test partition.
MIN_TEST_COUNT = 40
MIN_CLASS_SUPPORT = 8
# These class metrics are recomputed from the final router decision, including
# the train-derived always_escalate category policy. Raw score diagnostics stay
# available in the profile's split metrics only if a future trainer exports
# them separately; the gate must describe the shipped behavior.
MIN_RAW_BALANCED_ACCURACY = 0.65
MIN_RAW_ESCALATE_RECALL = 0.80
MIN_POLICY_FINAL_FAILURE_RECALL = 0.95
MIN_SLOW_FAILURE_PLANNED_RECALL = 0.90
MAX_UNSAFE_LOCAL_RATE = 0.02
MAX_PROJECTED_MISSES_19 = 2.0
METRIC_TOLERANCE = 5e-5


def _positive_int(profile, field):
    value = profile.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"router profile {field} must be a positive integer")
    return value


def _nonnegative_int(profile, field):
    value = profile.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(
            f"router profile {field} must be a non-negative integer")
    return value


def _finite_ratio(value, field):
    if isinstance(value, bool) or not isinstance(value, (int, float)) \
            or not math.isfinite(float(value)) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{field} must be finite in [0, 1]")
    return float(value)


def _metric_close(actual, expected, field):
    if abs(float(actual) - float(expected)) > METRIC_TOLERANCE:
        raise ValueError(
            f"serialized {field} does not match recomputed value: "
            f"{actual} != {expected}")


def _validate_confusion(test_metrics):
    confusion = test_metrics.get("confusion")
    if not isinstance(confusion, dict):
        raise ValueError("router profile metrics.test.confusion must be an object")
    values = {}
    for key in ("tp", "fp", "tn", "fn"):
        value = confusion.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(
                f"router profile metrics.test.confusion.{key} must be a "
                "non-negative integer")
        values[key] = value
    if sum(values.values()) != test_metrics["count"]:
        raise ValueError(
            "router profile metrics.test confusion does not sum to count")
    positive_support = values["tp"] + values["fn"]
    negative_support = values["tn"] + values["fp"]
    if positive_support < MIN_CLASS_SUPPORT or negative_support < MIN_CLASS_SUPPORT:
        raise ValueError(
            "router held-out test lacks class support: "
            f"escalate={positive_support} local_ok={negative_support} "
            f"minimum={MIN_CLASS_SUPPORT}")
    return values, positive_support, negative_support


def _excluded_task_ids(profile):
    exclusion = profile.get("exclusion")
    if not isinstance(exclusion, dict):
        raise ValueError("router profile exclusion diagnostics are missing")
    groups = exclusion.get("matchedGroups")
    if not isinstance(groups, list):
        raise ValueError("router profile exclusion.matchedGroups must be an array")
    result = set()
    for group in groups:
        if not isinstance(group, dict) or not isinstance(group.get("taskIds"), list):
            raise ValueError("router profile exclusion group taskIds are invalid")
        for task_id in group["taskIds"]:
            if not isinstance(task_id, str) or not task_id:
                raise ValueError("router profile exclusion task_id is invalid")
            if task_id in result:
                raise ValueError(f"duplicate excluded task_id {task_id!r}")
            result.add(task_id)
    if exclusion.get("excludedRecordCount") != len(result):
        raise ValueError(
            "router profile exclusion count does not match excluded task IDs")
    return result


def _validate_split_manifest(profile, labels_by_id):
    split = profile.get("split")
    task_ids = split.get("task_ids") if isinstance(split, dict) else None
    if not isinstance(task_ids, dict):
        raise ValueError("router profile split.task_ids is missing")
    partitions = {}
    for name in ("train", "calibration", "test"):
        values = task_ids.get(name)
        if not isinstance(values, list) or not all(
                isinstance(value, str) and value for value in values):
            raise ValueError(f"router profile split.task_ids.{name} is invalid")
        if len(values) != len(set(values)):
            raise ValueError(f"router profile split {name} contains duplicates")
        partitions[name] = set(values)
    if partitions["train"] & partitions["calibration"] \
            or partitions["train"] & partitions["test"] \
            or partitions["calibration"] & partitions["test"]:
        raise ValueError("router profile split task IDs overlap")
    excluded = _excluded_task_ids(profile)
    union = set().union(*partitions.values())
    expected = set(labels_by_id) - excluded
    if union != expected:
        raise ValueError(
            "router profile split/exclusion IDs do not cover label records")
    if excluded & union or excluded - set(labels_by_id):
        raise ValueError("router profile exclusion IDs are inconsistent")
    return partitions, excluded


def _validate_provenance(profile, labels, labels_path, tasks_path):
    provenance = profile.get("provenance")
    if not isinstance(provenance, dict) or provenance.get("schemaVersion") != 1:
        raise ValueError("router profile provenance schemaVersion must be 1")
    expected_labels_sha = canonical_records_sha256(labels)
    if provenance.get("labelRecordsSha256") != expected_labels_sha:
        raise ValueError("router profile labelRecordsSha256 mismatch")
    if provenance.get("labelRecordCount") != len(labels):
        raise ValueError("router profile provenance labelRecordCount mismatch")
    if provenance.get("requiredTasksSha256") != file_sha256(tasks_path):
        raise ValueError("router profile requiredTasksSha256 mismatch")
    run = provenance.get("labelRun")
    if run != labels[0].get("provenance"):
        raise ValueError("router profile labelRun provenance mismatch")
    current_solver = solver_bundle_sha256()
    if provenance.get("solverBundleSha256") != current_solver \
            or run.get("solver_bundle_sha256") != current_solver:
        raise ValueError("router profile solver provenance mismatch")
    current_labeler = file_sha256(ROOT / "eval/label_local.py")
    if provenance.get("labelerSha256") != current_labeler \
            or run.get("labeler_sha256") != current_labeler:
        raise ValueError("router profile labeler provenance mismatch")
    if run.get("task_manifest_sha256") != file_sha256(tasks_path):
        raise ValueError("label task-manifest provenance mismatch")
    source_files = provenance.get("solverFiles")
    if not isinstance(source_files, dict):
        raise ValueError("router profile solverFiles provenance is missing")
    for relative, expected_sha in source_files.items():
        path = ROOT / relative
        if not path.is_file() or file_sha256(path) != expected_sha:
            raise ValueError(f"router profile solver file provenance mismatch: {relative}")
    excluded_files = provenance.get("excludedTaskFiles")
    if not isinstance(excluded_files, list):
        raise ValueError("router profile excludedTaskFiles provenance is missing")
    for item in excluded_files:
        if not isinstance(item, dict):
            raise ValueError("router profile excluded task provenance is invalid")
        path = Path(str(item.get("path") or ""))
        if not path.is_absolute():
            path = ROOT / path
        if not path.is_file() or file_sha256(path) != item.get("sha256"):
            raise ValueError(
                f"router profile excluded task provenance mismatch: {path}")
    if provenance.get("profileSemanticSha256") != \
            profile_semantic_sha256(profile):
        raise ValueError("router profile semantic provenance hash mismatch")


def _recompute_test_evaluation(profile, test_records):
    scored = []
    for record in test_records:
        item = dict(record)
        item["score"] = score_compact_prompt(item["prompt"], profile)
        scored.append(item)
    threshold = float(profile["threshold"])
    policy = profile.get("categoryPolicy") or {
        "always_escalate": [], "trust_local": []}
    recomputed_metrics = classification_metrics(scored, threshold, policy)
    training = profile.get("training") or {}
    projection = simulate(
        scored, threshold, policy,
        esc_tokens=float(training.get("escalationTokens", 180.0)),
        esc_accuracy=float(training.get("escalationAccuracy", 0.95)),
        reactive_deadline_s=float(
            training.get("reactiveDeadlineSeconds", 25.0)),
        min_remote_window_s=float(
            training.get("minimumRemoteWindowSeconds", 8.0)),
    )

    safety = policy_safety_metrics(
        scored, threshold, policy,
        reactive_deadline_s=float(
            training.get("reactiveDeadlineSeconds", 25.0)),
        min_remote_window_s=float(
            training.get("minimumRemoteWindowSeconds", 8.0)),
    )
    return recomputed_metrics, projection, safety


def _validate_trained_profile(profile, label_count):
    """Validate fields that the runtime schema deliberately treats as metadata."""
    validate_compact_profile(profile)

    revision = str(profile.get("revision") or "").strip()
    if not revision or "pending" in revision.casefold():
        raise ValueError("router profile revision is missing or pending")
    weights = profile.get("weights")
    if not isinstance(weights, dict) or not weights:
        raise ValueError("router profile weights must be non-empty")

    trained = _positive_int(profile, "trainedExamples")
    total = _positive_int(profile, "totalExamples")
    excluded = _nonnegative_int(profile, "excludedExamples")
    if trained > total:
        raise ValueError("router profile trainedExamples exceeds totalExamples")
    if total + excluded != label_count:
        raise ValueError(
            "router profile label coverage mismatch: "
            f"totalExamples={total} excludedExamples={excluded} labels={label_count}")

    category_stats = profile.get("categoryStats")
    if not isinstance(category_stats, dict):
        raise ValueError("router profile categoryStats must be an object")
    missing_categories = sorted(set(CATEGORIES) - set(category_stats))
    if missing_categories:
        raise ValueError(
            f"router profile categoryStats missing categories: {missing_categories}")

    test_metrics = (profile.get("metrics") or {}).get("test")
    if not isinstance(test_metrics, dict):
        raise ValueError("router profile metrics.test must be an object")
    count = test_metrics.get("count")
    if isinstance(count, bool) or not isinstance(count, int) \
            or count < MIN_TEST_COUNT:
        raise ValueError(
            "router profile metrics.test.count must be at least "
            f"{MIN_TEST_COUNT}")
    for field in ("accuracy", "escalatePrecision", "escalateRecall",
                  "localOkPrecision", "localOkRecall"):
        _finite_ratio(
            test_metrics.get(field), f"router profile metrics.test.{field}")
    _, _, _ = _validate_confusion(test_metrics)
    balanced_accuracy = (
        float(test_metrics["escalateRecall"])
        + float(test_metrics["localOkRecall"])) / 2.0
    if balanced_accuracy < MIN_RAW_BALANCED_ACCURACY:
        raise ValueError(
            "router held-out balanced accuracy below safety gate: "
            f"{balanced_accuracy:.4f} < {MIN_RAW_BALANCED_ACCURACY:.4f}")
    if float(test_metrics["escalateRecall"]) < MIN_RAW_ESCALATE_RECALL:
        raise ValueError(
            "router held-out escalate recall below safety gate: "
            f"{test_metrics['escalateRecall']:.4f} < "
            f"{MIN_RAW_ESCALATE_RECALL:.4f}")
    return revision, trained


def check_router_artifacts(core_path, demo_path, labels_path, tasks_path):
    """Validate artifacts and return a small success report."""
    core = Path(core_path)
    demo = Path(demo_path)
    core_bytes = core.read_bytes()
    demo_bytes = demo.read_bytes()
    if not core_bytes or not demo_bytes:
        raise ValueError("router profile artifacts must be non-empty")

    core_sha = hashlib.sha256(core_bytes).hexdigest()
    demo_sha = hashlib.sha256(demo_bytes).hexdigest()
    if core_sha != demo_sha or core_bytes != demo_bytes:
        raise ValueError(
            "core/demo router profiles differ: "
            f"core_sha256={core_sha} demo_sha256={demo_sha}")

    try:
        profile = json.loads(core_bytes)
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError(f"router profile is not valid UTF-8 JSON: {exc}") from exc
    if not isinstance(profile, dict):
        raise ValueError("router profile must be a JSON object")

    labels = load_label_records([str(labels_path)])
    covered = validate_required_task_coverage(labels, tasks_path)
    revision, trained = _validate_trained_profile(profile, len(labels))
    _validate_provenance(profile, labels, labels_path, tasks_path)

    labels_by_id = {record["task_id"]: record for record in labels}
    partitions, excluded = _validate_split_manifest(profile, labels_by_id)
    test_records = [labels_by_id[task_id]
                    for task_id in sorted(partitions["test"])]
    recomputed_metrics, recomputed_projection, recomputed_safety = \
        _recompute_test_evaluation(profile, test_records)
    serialized_metrics = profile["metrics"]["test"]
    for field in ("accuracy", "escalatePrecision", "escalateRecall",
                  "localOkPrecision", "localOkRecall"):
        _metric_close(
            serialized_metrics[field], recomputed_metrics[field],
            f"metrics.test.{field}")
    if serialized_metrics.get("count") != recomputed_metrics["count"] \
            or serialized_metrics.get("confusion") != recomputed_metrics["confusion"]:
        raise ValueError(
            "serialized metrics.test count/confusion do not match recomputation")

    serialized_projection = profile.get("testProjection19")
    if not isinstance(serialized_projection, dict):
        raise ValueError("router profile testProjection19 is missing")
    for field in ("expected_misses", "expected_escalations", "expected_tokens",
                  "local_share", "projected_accuracy", "planned_escalations",
                  "reactive_escalations", "reactive_at_risk",
                  "expected_fallback_misses", "slow_failure_recall"):
        if field not in serialized_projection:
            raise ValueError(f"router profile testProjection19.{field} is missing")
        _metric_close(
            serialized_projection[field], recomputed_projection[field],
            f"testProjection19.{field}")
    if recomputed_projection["expected_misses"] > MAX_PROJECTED_MISSES_19:
        raise ValueError(
            "router held-out expected misses exceed safety gate: "
            f"{recomputed_projection['expected_misses']} > "
            f"{MAX_PROJECTED_MISSES_19}")

    serialized_safety = (profile.get("safetyEvaluation") or {}).get("test")
    if not isinstance(serialized_safety, dict):
        raise ValueError("router profile safetyEvaluation.test is missing")
    if serialized_safety != recomputed_safety:
        raise ValueError(
            "serialized safetyEvaluation.test does not match recomputation")
    if recomputed_safety["failureFinalEscalationRecall"] < \
            MIN_POLICY_FINAL_FAILURE_RECALL:
        raise ValueError(
            "router policy final failure recall below safety gate: "
            f"{recomputed_safety['failureFinalEscalationRecall']:.4f} < "
            f"{MIN_POLICY_FINAL_FAILURE_RECALL:.4f}")
    if recomputed_safety["slowFailurePlannedRecall"] < \
            MIN_SLOW_FAILURE_PLANNED_RECALL:
        raise ValueError(
            "router slow-failure planned recall below safety gate: "
            f"{recomputed_safety['slowFailurePlannedRecall']:.4f} < "
            f"{MIN_SLOW_FAILURE_PLANNED_RECALL:.4f}")
    if recomputed_safety["unsafeLocalRate"] > MAX_UNSAFE_LOCAL_RATE:
        raise ValueError(
            "router unsafe-local rate exceeds safety gate: "
            f"{recomputed_safety['unsafeLocalRate']:.4f} > "
            f"{MAX_UNSAFE_LOCAL_RATE:.4f}")

    train_ids = [labels_by_id[task_id] for task_id in partitions["train"]]
    expected_revision = _profile_revision(
        profile, [record["task_id"] for record in train_ids])
    if revision != expected_revision:
        raise ValueError(
            f"router profile revision mismatch: {revision} != {expected_revision}")
    return {
        "sha256": core_sha,
        "revision": revision,
        "labels": covered,
        "trained": trained,
        "excluded": len(excluded),
        "test": len(test_records),
        "balanced_accuracy": round((
            recomputed_metrics["escalateRecall"]
            + recomputed_metrics["localOkRecall"]) / 2.0, 4),
        "slow_failure_recall": recomputed_safety[
            "slowFailurePlannedRecall"],
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--core", default="router_model/compact_router.json")
    parser.add_argument(
        "--demo", default="demo/src/data/router-profile.json")
    parser.add_argument("--labels", required=True)
    parser.add_argument("--require-tasks", required=True)
    args = parser.parse_args(argv)
    try:
        report = check_router_artifacts(
            args.core, args.demo, args.labels, args.require_tasks)
    except (OSError, ValueError) as exc:
        print(f"router artifact check failed: {exc}", file=sys.stderr)
        return 1
    print(
        "router artifact check ok: "
        f"revision={report['revision']} labels={report['labels']} "
        f"trained={report['trained']} excluded={report['excluded']} "
        f"test={report['test']} balanced_acc={report['balanced_accuracy']:.4f} "
        f"slow_recall={report['slow_failure_recall']:.4f} "
        f"sha256={report['sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
