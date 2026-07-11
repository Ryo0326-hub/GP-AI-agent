#!/usr/bin/env python3
"""Train the zero-dependency hashed-logistic query router.

The exported profile is intentionally shared by the Python agent and the
Next.js browser demo.  Features are sparse hashed unigrams/bigrams plus the
same category/shape hints already used by ``demo/src/lib/local-router.ts``.
Training uses deterministic full-batch Adam implemented with the standard
library; no torch, transformers, or model download is required.

Example:
  python3 router/train_compact_router.py \
    --labels data/labels_15b_2cpu.jsonl \
    --exclude-tasks test_input/tasks.json test_input_19/tasks.json \
    --out router_model/compact_router.json \
    --demo-out demo/src/data/router-profile.json
"""
import argparse
import hashlib
import json
import math
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from router import (  # noqa: E402
    compact_feature_values,
    score_compact_prompt,
    validate_compact_profile,
)
from classifier import CATEGORIES, classify  # noqa: E402
from threshold import (MIN_REMOTE_WINDOW_S, MIN_SLOW_FAILURE_RECALL,
                       REACTIVE_DEADLINE_S, derive_policy, pick_threshold,
                       policy_safety_metrics, simulate)  # noqa: E402
from train_router import (  # noqa: E402
    _template_similarity,
    canonical_prompt_template,
    group_prompt_templates,
    grouped_train_calibration_test_split,
    summarize_training_categories,
)

LABEL_TO_ID = {"local_ok": 0, "escalate": 1}
PROVENANCE_HASH_FIELDS = (
    "gguf_sha256",
    "solver_bundle_sha256",
    "labeler_sha256",
    "task_manifest_sha256",
    "expected_answers_sha256",
)
SOLVER_PROVENANCE_FILES = (
    "app/main.py",
    "app/local_model.py",
    "app/classifier.py",
    "app/prompts.py",
    "app/verify.py",
    "app/utils.py",
)


def load_label_records(paths):
    """Read and strictly validate one or more empirical label JSONL files."""
    records = []
    for raw_path in paths:
        path = Path(raw_path)
        with path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except ValueError as exc:
                    raise ValueError(
                        f"{path}:{line_number}: invalid JSON") from exc
                if not isinstance(record, dict):
                    raise ValueError(
                        f"{path}:{line_number}: record must be an object")
                if not isinstance(record.get("prompt"), str) \
                        or not record["prompt"].strip():
                    raise ValueError(
                        f"{path}:{line_number}: prompt must be non-empty")
                if record.get("label") not in LABEL_TO_ID:
                    raise ValueError(
                        f"{path}:{line_number}: label must be local_ok or "
                        "escalate")
                record = dict(record)
                task_id = record.get("task_id")
                if not isinstance(task_id, str) or not task_id.strip() \
                        or task_id != task_id.strip():
                    raise ValueError(
                        f"{path}:{line_number}: task_id must be trimmed and "
                        "non-empty")
                for field in ("correct", "verified"):
                    if not isinstance(record.get(field), bool):
                        raise ValueError(
                            f"{path}:{line_number}: {field} must be boolean")
                if record["label"] != (
                        "local_ok" if record["correct"] else "escalate"):
                    raise ValueError(
                        f"{path}:{line_number}: label/correct mismatch")
                category = record.get("category")
                if category not in CATEGORIES:
                    raise ValueError(
                        f"{path}:{line_number}: category is invalid")
                runtime_category = classify(record["prompt"])
                if category != runtime_category:
                    raise ValueError(
                        f"{path}:{line_number}: category {category!r} does not "
                        f"match runtime classifier {runtime_category!r}")
                dataset_category = record.get("dataset_category")
                if dataset_category not in CATEGORIES:
                    raise ValueError(
                        f"{path}:{line_number}: dataset_category is invalid")
                latency = record.get("latency_s")
                if isinstance(latency, bool) or not isinstance(
                        latency, (int, float)) or not math.isfinite(float(latency)) \
                        or latency < 0:
                    raise ValueError(
                        f"{path}:{line_number}: latency_s must be finite and "
                        "non-negative")
                completion = record.get("completion_tokens")
                if isinstance(completion, bool) or not isinstance(completion, int) \
                        or completion < 0:
                    raise ValueError(
                        f"{path}:{line_number}: completion_tokens must be a "
                        "non-negative integer")
                model = record.get("model")
                if not isinstance(model, str) or not model.strip():
                    raise ValueError(
                        f"{path}:{line_number}: model tag must be non-empty")
                provenance = record.get("provenance")
                if not isinstance(provenance, dict) \
                        or provenance.get("schema_version") != 1:
                    raise ValueError(
                        f"{path}:{line_number}: provenance schema_version must be 1")
                for field in PROVENANCE_HASH_FIELDS:
                    value = provenance.get(field)
                    if not isinstance(value, str) or not re.fullmatch(
                            r"[0-9a-f]{64}", value):
                        raise ValueError(
                            f"{path}:{line_number}: provenance.{field} must be "
                            "a lowercase SHA-256")
                budget = provenance.get("budget_seconds")
                if isinstance(budget, bool) or not isinstance(
                        budget, (int, float)) or not math.isfinite(float(budget)) \
                        or budget <= 0:
                    raise ValueError(
                        f"{path}:{line_number}: provenance.budget_seconds must "
                        "be finite and positive")
                threads = provenance.get("threads")
                if isinstance(threads, bool) or not isinstance(threads, int) \
                        or threads <= 0:
                    raise ValueError(
                        f"{path}:{line_number}: provenance.threads must be a "
                        "positive integer")
                records.append(record)
    if not records:
        raise ValueError("no label records loaded")
    provenance_values = {
        json.dumps(record["provenance"], sort_keys=True, separators=(",", ":"))
        for record in records
    }
    if len(provenance_values) != 1:
        raise ValueError(
            "label records have mixed provenance; model, solver, manifest, "
            "budget, and thread settings must match")
    return records


def normalized_prompt(value):
    """Canonical key used only for explicit benchmark-prompt exclusion."""
    text = unicodedata.normalize("NFKC", str(value or ""))
    return re.sub(r"\s+", " ", text).strip().casefold()


def _sha256_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path):
    return _sha256_bytes(Path(path).read_bytes())


def canonical_records_sha256(records):
    """Path/order-independent digest of full empirical label records."""
    ordered = sorted((dict(record) for record in records),
                     key=lambda record: record["task_id"])
    payload = json.dumps(
        ordered, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return _sha256_bytes(payload)


def solver_bundle_sha256(root=ROOT):
    """Digest the local-solving source whose outcomes the labels describe."""
    chunks = []
    for relative in sorted(SOLVER_PROVENANCE_FILES):
        digest = file_sha256(Path(root) / relative)
        chunks.append(f"{relative}\0{digest}\n")
    return _sha256_bytes("".join(chunks).encode("utf-8"))


def profile_semantic_sha256(profile):
    """Digest a profile excluding only its self-referential digest field."""
    clone = json.loads(json.dumps(profile))
    provenance = clone.get("provenance")
    if isinstance(provenance, dict):
        provenance.pop("profileSemanticSha256", None)
    payload = json.dumps(
        clone, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return _sha256_bytes(payload)


def _task_scope(task):
    explicit = str(task.get("dataset_category") or "").strip()
    if explicit:
        return explicit
    task_id = str(task.get("task_id") or "")
    prefix = task_id.rsplit("_", 1)[0]
    if prefix in CATEGORIES:
        return prefix
    return classify(str(task.get("prompt") or ""))


def _load_tasks_payload(raw_path):
    path = Path(raw_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("tasks")
    if not isinstance(payload, list):
        raise ValueError(f"{path}: tasks JSON must be an array")
    tasks = []
    for index, task in enumerate(payload):
        if not isinstance(task, dict) or not isinstance(
                task.get("prompt"), str):
            raise ValueError(
                f"{path}: task {index} must contain a string prompt")
        item = dict(task)
        item["dataset_category"] = _task_scope(item)
        item["category"] = classify(item["prompt"])
        tasks.append(item)
    return tasks


def validate_required_task_coverage(records, tasks_path):
    """Fail unless labels cover exactly the required task manifest.

    Coverage is checked before benchmark-prompt exclusion: every generated
    training task must have one label, even when a matching public rehearsal
    prompt is subsequently removed from model fitting. Prompt equality catches
    stale labels that happen to reuse the same task IDs after dataset changes.
    """
    path = Path(tasks_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("tasks")
    if not isinstance(payload, list):
        raise ValueError(f"{path}: required tasks JSON must be an array")

    required = {}
    for index, task in enumerate(payload):
        if not isinstance(task, dict):
            raise ValueError(f"{path}: task {index} must be an object")
        task_id = task.get("task_id")
        prompt = task.get("prompt")
        if not isinstance(task_id, str) or not task_id.strip() \
                or task_id != task_id.strip():
            raise ValueError(
                f"{path}: task {index} must contain a trimmed, non-empty "
                "string task_id")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(
                f"{path}: task {index} must contain a non-empty string prompt")
        if task_id in required:
            raise ValueError(f"{path}: duplicate required task_id {task_id!r}")
        required[task_id] = normalized_prompt(prompt)

    labeled = {}
    duplicates = []
    for index, record in enumerate(records):
        task_id = record.get("task_id")
        if not isinstance(task_id, str) or not task_id.strip() \
                or task_id != task_id.strip():
            raise ValueError(
                f"label record {index} must contain a trimmed, non-empty "
                "string task_id")
        if task_id in labeled:
            duplicates.append(task_id)
        else:
            labeled[task_id] = normalized_prompt(record.get("prompt"))

    missing = sorted(set(required) - set(labeled))
    extras = sorted(set(labeled) - set(required))
    mismatched = sorted(
        task_id for task_id in set(required) & set(labeled)
        if required[task_id] != labeled[task_id]
    )
    if duplicates or missing or extras or mismatched:
        def preview(values):
            values = sorted(set(values))
            suffix = "..." if len(values) > 10 else ""
            return f"{values[:10]}{suffix}"

        raise ValueError(
            "required task coverage mismatch: "
            f"required={len(required)} labeled={len(records)} unique={len(labeled)} "
            f"missing={preview(missing)} extras={preview(extras)} "
            f"duplicates={preview(duplicates)} "
            f"prompt_mismatches={preview(mismatched)}"
        )
    return len(required)


def build_training_provenance(records, label_paths, required_tasks_path,
                              excluded_task_paths):
    """Bind the profile to empirical labels, manifests, and solver source."""
    run = dict(records[0]["provenance"])
    current_solver = solver_bundle_sha256()
    if run["solver_bundle_sha256"] != current_solver:
        raise ValueError(
            "label provenance solver_bundle_sha256 does not match current "
            "local-solving source; regenerate labels")
    current_labeler = file_sha256(ROOT / "eval/label_local.py")
    if run["labeler_sha256"] != current_labeler:
        raise ValueError(
            "label provenance labeler_sha256 does not match current grading "
            "code; regenerate labels")
    if required_tasks_path:
        required_sha = file_sha256(required_tasks_path)
        if run["task_manifest_sha256"] != required_sha:
            raise ValueError(
                "label provenance task_manifest_sha256 does not match "
                "--require-tasks")
    else:
        required_sha = run["task_manifest_sha256"]

    solver_files = {
        relative: file_sha256(ROOT / relative)
        for relative in sorted(SOLVER_PROVENANCE_FILES)
    }
    return {
        "schemaVersion": 1,
        "labelRecordsSha256": canonical_records_sha256(records),
        "labelRecordCount": len(records),
        "labelFiles": [
            {"path": str(path), "sha256": file_sha256(path)}
            for path in label_paths
        ],
        "requiredTasksSha256": required_sha,
        "requiredTasksPath": str(required_tasks_path or ""),
        "excludedTaskFiles": [
            {"path": str(path), "sha256": file_sha256(path)}
            for path in excluded_task_paths
        ],
        "labelRun": run,
        "solverBundleSha256": current_solver,
        "labelerSha256": current_labeler,
        "solverFiles": solver_files,
    }


def load_excluded_prompts(paths):
    """Read prompt strings from supplied tasks.json files."""
    prompts = set()
    for raw_path in paths or ():
        path = Path(raw_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = payload.get("tasks")
        if not isinstance(payload, list):
            raise ValueError(f"{path}: tasks JSON must be an array")
        for index, task in enumerate(payload):
            if not isinstance(task, dict) or not isinstance(
                    task.get("prompt"), str):
                raise ValueError(
                    f"{path}: task {index} must contain a string prompt")
            prompts.add(normalized_prompt(task["prompt"]))
    return prompts


def load_excluded_tasks(paths):
    """Load public task manifests with canonicalization scope metadata."""
    tasks = []
    for raw_path in paths or ():
        for task in _load_tasks_payload(raw_path):
            task["source_file"] = str(raw_path)
            tasks.append(task)
    return tasks


def exclude_prompt_matches(records, excluded_prompts):
    """Return (kept, excluded) without mutating the caller's records."""
    kept, excluded = [], []
    for record in records:
        target = excluded if normalized_prompt(record["prompt"]) \
            in excluded_prompts else kept
        target.append(record)
    return kept, excluded


def _template_group_digest(group):
    stable = sorted({
        canonical_prompt_template(record)
        for record in group
    })
    return _sha256_bytes(json.dumps(
        stable, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8"))[:16]


def exclude_template_groups(records, excluded_tasks, similarity_threshold=0.82):
    """Exclude complete label groups matching public prompts/templates.

    A single public sibling removes the entire generated template group. This
    prevents an exact exclusion such as "Write ..." leaving an "Implement ..."
    sibling available for fitting or internal evaluation.
    """
    groups = group_prompt_templates(records, similarity_threshold)
    public = []
    for task in excluded_tasks:
        public.append({
            "task_id": str(task.get("task_id") or ""),
            "normalized": normalized_prompt(task["prompt"]),
            "canonical": canonical_prompt_template(task),
            "scope": _task_scope(task),
            "source_file": str(task.get("source_file") or ""),
        })

    kept, excluded = [], []
    matches = []
    for group in groups:
        best = None
        for record in group:
            record_normalized = normalized_prompt(record["prompt"])
            record_canonical = canonical_prompt_template(record)
            record_scope = _task_scope(record)
            for task in public:
                exact = record_normalized == task["normalized"]
                similarity = 1.0 if exact else 0.0
                if not exact and record_scope == task["scope"]:
                    similarity = _template_similarity(
                        record_canonical, task["canonical"])
                if exact or similarity >= similarity_threshold:
                    candidate = {
                        "matchType": "exact" if exact else "template",
                        "similarity": round(similarity, 6),
                        "publicTaskId": task["task_id"],
                        "publicTaskDigest": _sha256_bytes(
                            task["canonical"].encode("utf-8"))[:16],
                        "sourceFile": task["source_file"],
                    }
                    if best is None or candidate["similarity"] > best["similarity"]:
                        best = candidate
        if best is None:
            kept.extend(group)
            continue
        excluded.extend(group)
        matches.append({
            "groupDigest": _template_group_digest(group),
            "recordCount": len(group),
            "taskIds": sorted(str(record["task_id"]) for record in group),
            **best,
        })

    diagnostics = {
        "strategy": "grouped_public_template_v1",
        "similarityThreshold": similarity_threshold,
        "publicPromptCount": len(public),
        "inputGroupCount": len(groups),
        "excludedGroupCount": len(matches),
        "excludedRecordCount": len(excluded),
        "exactMatchedGroupCount": sum(
            match["matchType"] == "exact" for match in matches),
        "templateMatchedGroupCount": sum(
            match["matchType"] == "template" for match in matches),
        "matchedGroups": sorted(matches, key=lambda item: item["groupDigest"]),
    }
    return kept, excluded, diagnostics


def _sigmoid(logit):
    bounded = max(-30.0, min(30.0, logit))
    return 1.0 / (1.0 + math.exp(-bounded))


def _training_examples(records, dimension):
    examples = []
    for record in records:
        # Runtime category classification is deliberately performed by
        # score_compact_prompt.  Importing the same classifier here keeps the
        # learned category feature identical to inference.
        from router import _classify_for_compact  # local/private twin
        category = _classify_for_compact(record["prompt"])
        examples.append((
            compact_feature_values(record["prompt"], category, dimension),
            LABEL_TO_ID[record["label"]],
        ))
    return examples


def train_logistic(records, dimension=2048, epochs=350, learning_rate=0.04,
                   l2=0.0002):
    """Fit weighted sparse logistic regression with deterministic Adam."""
    if not records:
        raise ValueError("training split is empty")
    if dimension <= 0:
        raise ValueError("dimension must be positive")
    if epochs <= 0 or learning_rate <= 0 or l2 < 0:
        raise ValueError("epochs/lr must be positive and l2 non-negative")
    examples = _training_examples(records, dimension)
    positives = sum(label for _, label in examples)
    negatives = len(examples) - positives
    if not positives or not negatives:
        raise ValueError("training split must contain both route labels")
    positive_weight = negatives / positives
    normalizer = negatives + positives * positive_weight

    weights = [0.0] * dimension
    first_moment = [0.0] * dimension
    second_moment = [0.0] * dimension
    bias = bias_m = bias_v = 0.0
    beta1, beta2, epsilon = 0.9, 0.999, 1e-8

    for step in range(1, epochs + 1):
        gradient = {}
        bias_gradient = 0.0
        for features, label in examples:
            logit = bias + sum(weights[index] * value
                               for index, value in features.items())
            sample_weight = positive_weight if label else 1.0
            residual = sample_weight * (_sigmoid(logit) - label)
            bias_gradient += residual
            for index, value in features.items():
                gradient[index] = gradient.get(index, 0.0) + residual * value

        bias_gradient /= normalizer
        bias_m = beta1 * bias_m + (1.0 - beta1) * bias_gradient
        bias_v = beta2 * bias_v + (1.0 - beta2) * bias_gradient ** 2
        bias_m_hat = bias_m / (1.0 - beta1 ** step)
        bias_v_hat = bias_v / (1.0 - beta2 ** step)
        bias -= learning_rate * bias_m_hat / (math.sqrt(bias_v_hat) + epsilon)

        for index in range(dimension):
            value = gradient.get(index, 0.0) / normalizer + l2 * weights[index]
            first_moment[index] = beta1 * first_moment[index] \
                + (1.0 - beta1) * value
            second_moment[index] = beta2 * second_moment[index] \
                + (1.0 - beta2) * value ** 2
            m_hat = first_moment[index] / (1.0 - beta1 ** step)
            v_hat = second_moment[index] / (1.0 - beta2 ** step)
            weights[index] -= learning_rate * m_hat \
                / (math.sqrt(v_hat) + epsilon)

    sparse_weights = {
        str(index): round(weight, 10)
        for index, weight in enumerate(weights)
        if abs(weight) >= 1e-9
    }
    return round(bias, 10), sparse_weights


def score_records(records, profile):
    """Copy records and attach compact P(escalate) scores."""
    scored = []
    for record in records:
        item = dict(record)
        item["score"] = score_compact_prompt(item["prompt"], profile)
        scored.append(item)
    return scored


def classification_metrics(records, threshold, policy=None):
    """Return metrics for the final router decision and confusion matrix.

    The shipped router has two explicit train-derived controls in addition to
    the learned score: ``always_escalate`` category overrides and the optional
    verification-gated ``trust_local`` policy.  Reporting only the raw score
    would misstate the behavior that actually runs in the container, so the
    serialized class metrics include the planned category override.
    """
    always_escalate = set((policy or {}).get("always_escalate") or ())
    tp = fp = tn = fn = 0
    for record in records:
        predicted = 1 if (
            record.get("category") in always_escalate
            or float(record["score"]) >= threshold
        ) else 0
        actual = LABEL_TO_ID[record["label"]]
        if predicted and actual:
            tp += 1
        elif predicted:
            fp += 1
        elif actual:
            fn += 1
        else:
            tn += 1

    def ratio(numerator, denominator):
        return round(numerator / denominator, 4) if denominator else 0.0

    total = tp + fp + tn + fn
    return {
        "count": total,
        "accuracy": ratio(tp + tn, total),
        "escalatePrecision": ratio(tp, tp + fp),
        "escalateRecall": ratio(tp, tp + fn),
        "localOkPrecision": ratio(tn, tn + fn),
        "localOkRecall": ratio(tn, tn + fp),
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    }


def derive_training_runtime_config(train_records):
    """Derive every runtime policy/cost field from the training split only."""
    if not train_records:
        raise ValueError("training split is empty")
    policy = derive_policy(train_records)
    category_stats = summarize_training_categories(train_records)
    expected_tokens = {
        category: stats["p90_completion_tokens"]
        for category, stats in category_stats.items()
    }
    return policy, expected_tokens, category_stats


def _profile_revision(profile, record_ids):
    stable = {
        "modelType": profile["modelType"],
        "dimension": profile["dimension"],
        "bias": profile["bias"],
        "threshold": profile["threshold"],
        "weights": profile["weights"],
        "categoryPolicy": profile.get("categoryPolicy", {}),
        "expectedCompletionTokens": profile.get(
            "expectedCompletionTokens", {}),
        "expectedLocalLatencySeconds": profile.get(
            "expectedLocalLatencySeconds", {}),
        "split": profile.get("split", {}),
        "exclusion": profile.get("exclusion", {}),
        "provenance": {
            key: value for key, value in profile.get("provenance", {}).items()
            if key != "profileSemanticSha256"
        },
        "recordIds": sorted(record_ids),
    }
    digest = hashlib.sha256(json.dumps(
        stable, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:12]
    return f"compact-{digest}"


def build_profile(train_records, calibration_records, test_records, *,
                  dimension=2048, epochs=350, learning_rate=0.04,
                  l2=0.0002, max_expected_misses=1.0,
                  esc_tokens=180.0, esc_accuracy=0.95,
                  split_diagnostics=None, excluded_records=None,
                  exclusion_diagnostics=None, provenance=None,
                  labels=None, exclude_tasks=None, seed=7):
    """Fit, calibrate, evaluate once, and return an exportable profile."""
    bias, weights = train_logistic(
        train_records, dimension=dimension, epochs=epochs,
        learning_rate=learning_rate, l2=l2,
    )
    profile = {
        "schemaVersion": 1,
        "revision": "pending",
        "modelType": "hashed-logistic-v1",
        "dimension": dimension,
        "bias": bias,
        "threshold": 0.5,
        "trainedExamples": len(train_records),
        "totalExamples": (
            len(train_records) + len(calibration_records) + len(test_records)
        ),
        "excludedExamples": len(excluded_records or ()),
        "weights": weights,
        "metrics": {},
    }
    category_policy, expected_tokens, category_stats = \
        derive_training_runtime_config(train_records)
    expected_latency = {
        category: stats["p90_latency_s"]
        for category, stats in category_stats.items()
    }
    profile.update({
        "categoryPolicy": category_policy,
        "categoryPolicySource": "train",
        "expectedCompletionTokens": expected_tokens,
        "expectedCompletionTokensSource": "train",
        "expectedLocalLatencySeconds": expected_latency,
        "expectedLocalLatencySecondsSource": "train",
        "categoryStats": category_stats,
        "categoryStatsSource": "train",
    })
    # Validate rounded export weights before calibration so Python and browser
    # metrics describe the actual artifact, not higher-precision trainer state.
    validated = validate_compact_profile(profile)
    calibration_scored = score_records(calibration_records, validated)
    threshold, calibration_projection = pick_threshold(
        calibration_scored, category_policy,
        max_expected_misses=max_expected_misses,
        esc_tokens=esc_tokens, esc_accuracy=esc_accuracy,
    )
    profile["threshold"] = threshold
    validated = validate_compact_profile(profile)
    train_scored = score_records(train_records, validated)
    calibration_scored = score_records(calibration_records, validated)
    # The final test split is first scored only after weights and threshold are
    # frozen.  It therefore remains a true one-shot evaluation artifact.
    test_scored = score_records(test_records, validated)
    split_metrics = {
        "train": classification_metrics(
            train_scored, threshold, category_policy),
        "calibration": classification_metrics(
            calibration_scored, threshold, category_policy),
        "test": classification_metrics(
            test_scored, threshold, category_policy),
    }
    profile["metrics"] = {
        **{
            key: value for key, value in split_metrics["test"].items()
            if key != "count"
        },
        "train": split_metrics["train"],
        "calibration": split_metrics["calibration"],
        "test": split_metrics["test"],
    }
    profile["calibrationProjection19"] = calibration_projection
    profile["testProjection19"] = simulate(
        test_scored, threshold, category_policy,
        esc_tokens=esc_tokens, esc_accuracy=esc_accuracy,
    )
    profile["safetyEvaluation"] = {
        "train": policy_safety_metrics(
            train_scored, threshold, category_policy),
        "calibration": policy_safety_metrics(
            calibration_scored, threshold, category_policy),
        "test": policy_safety_metrics(
            test_scored, threshold, category_policy),
    }
    profile["split"] = split_diagnostics or {}
    profile["exclusion"] = exclusion_diagnostics or {}
    profile["provenance"] = dict(provenance or {})
    profile["training"] = {
        "algorithm": "deterministic-full-batch-adam",
        "labels": list(labels or ()),
        "excludedTaskFiles": list(exclude_tasks or ()),
        "epochs": epochs,
        "learningRate": learning_rate,
        "l2": l2,
        "seed": seed,
        "thresholdSelection": "router.threshold.pick_threshold",
        "thresholdPolicy": "frozen_train_category_policy",
        "maxExpectedMissesPer19": max_expected_misses,
        "escalationTokens": esc_tokens,
        "escalationAccuracy": esc_accuracy,
        "reactiveDeadlineSeconds": REACTIVE_DEADLINE_S,
        "minimumRemoteWindowSeconds": MIN_REMOTE_WINDOW_S,
        "minimumSlowFailureRecall": MIN_SLOW_FAILURE_RECALL,
    }
    ids = [str(record.get("task_id") or record["prompt"])
           for record in train_records]
    profile["revision"] = _profile_revision(profile, ids)
    profile["provenance"]["profileSemanticSha256"] = \
        profile_semantic_sha256(profile)
    return profile


def write_profile(profile, paths):
    """Write byte-identical profile JSON to every requested destination."""
    payload = json.dumps(profile, indent=2, sort_keys=True) + "\n"
    for raw_path in paths:
        path = Path(raw_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")


def validate_split_class_support(name, records):
    labels = {record["label"] for record in records}
    missing = sorted(set(LABEL_TO_ID) - labels)
    if missing:
        raise ValueError(
            f"{name} split is missing route labels {missing}; collect more "
            "independent templates instead of evaluating a one-class split")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument(
        "--require-tasks",
        help=("tasks.json manifest that labels must cover exactly before "
              "benchmark exclusions are applied"),
    )
    parser.add_argument(
        "--exclude-tasks", nargs="*", default=[],
        help="tasks.json files whose prompts must be excluded before split",
    )
    parser.add_argument(
        "--out", default="router_model/compact_router.json",
        help="core runtime profile path",
    )
    parser.add_argument(
        "--demo-out", default="demo/src/data/router-profile.json",
        help="browser demo profile path (set empty to skip)",
    )
    parser.add_argument("--dimension", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=350)
    parser.add_argument("--learning-rate", type=float, default=0.04)
    parser.add_argument("--l2", type=float, default=0.0002)
    parser.add_argument("--calibration-frac", type=float, default=0.20)
    parser.add_argument("--test-frac", type=float, default=0.20)
    parser.add_argument("--template-similarity", type=float, default=0.82)
    parser.add_argument("--max-expected-misses", type=float, default=1.0)
    parser.add_argument("--esc-tokens", type=float, default=180.0)
    parser.add_argument("--esc-accuracy", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)

    records = load_label_records(args.labels)
    if args.require_tasks:
        covered = validate_required_task_coverage(records, args.require_tasks)
        print(f"required label coverage: {covered}/{covered}")
    provenance = build_training_provenance(
        records, args.labels, args.require_tasks, args.exclude_tasks)
    excluded_tasks = load_excluded_tasks(args.exclude_tasks)
    records, excluded_records, exclusion_diagnostics = \
        exclude_template_groups(
            records, excluded_tasks,
            similarity_threshold=args.template_similarity)
    exclusion_diagnostics["sourceFiles"] = [
        {"path": str(path), "sha256": file_sha256(path)}
        for path in args.exclude_tasks
    ]
    if not records:
        raise ValueError("benchmark exclusion removed every label record")
    train, calibration, test, diagnostics = \
        grouped_train_calibration_test_split(
            records,
            calibration_frac=args.calibration_frac,
            test_frac=args.test_frac,
            seed=args.seed,
            similarity_threshold=args.template_similarity,
            return_diagnostics=True,
        )
    if not train or not calibration or not test:
        raise ValueError(
            "grouped split requires non-empty train, calibration, and test")
    for split_name, split_records in (
            ("train", train), ("calibration", calibration), ("test", test)):
        validate_split_class_support(split_name, split_records)

    profile = build_profile(
        train, calibration, test,
        dimension=args.dimension,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2=args.l2,
        max_expected_misses=args.max_expected_misses,
        esc_tokens=args.esc_tokens,
        esc_accuracy=args.esc_accuracy,
        split_diagnostics=diagnostics,
        excluded_records=excluded_records,
        exclusion_diagnostics=exclusion_diagnostics,
        provenance=provenance,
        labels=args.labels,
        exclude_tasks=args.exclude_tasks,
        seed=args.seed,
    )
    destinations = [args.out] + ([args.demo_out] if args.demo_out else [])
    write_profile(profile, destinations)

    metrics = profile["metrics"]
    print(f"records: {len(records)} kept, {len(excluded_records)} excluded")
    print("split:", diagnostics["record_counts"])
    print(f"threshold: {profile['threshold']:.6f}")
    for split_name in ("train", "calibration", "test"):
        current = metrics[split_name]
        print(
            f"{split_name}: accuracy={current['accuracy']:.4f} "
            f"escalate P/R={current['escalatePrecision']:.4f}/"
            f"{current['escalateRecall']:.4f} "
            f"local_ok P/R={current['localOkPrecision']:.4f}/"
            f"{current['localOkRecall']:.4f}")
    print("revision:", profile["revision"])
    print("wrote:", ", ".join(destinations))
    return profile


if __name__ == "__main__":
    main()
