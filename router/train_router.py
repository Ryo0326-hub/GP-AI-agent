#!/usr/bin/env python3
"""Fine-tune DistilBERT as a binary local_ok/escalate router on our labels.

Adapted from github.com/Stephen-Kimoi/fine-tune-llm-query-router-amd, with one
critical change: instead of routing between two Fireworks models, the labels
say whether OUR local pipeline (Qwen GGUF + deterministic verification)
produced a correct answer. Predicting "escalate" from prompt text alone is
what lets the orchestrator dispatch hopeless tasks to Fireworks immediately
and keep everything else at zero tokens.

Works standalone (notebook-friendly: call main([...]) with args) on CUDA,
ROCm (AMD MI300X: the ROCm torch build exposes the GPU as "cuda"), MPS, or CPU.

Usage:
  python3 router/train_router.py --labels data/labels_3b.jsonl \
      --stats data/category_stats_3b.json --out router_model --device auto

Outputs into --out:
  config.json / model.safetensors / tokenizer files  (fp32 checkpoint;
      the runtime applies int8 dynamic quantization at load)
  router_config.json  threshold, per-category policy, expected tokens,
      calibration metrics and an untouched test-set projection
"""
import argparse
from collections import Counter, defaultdict
from difflib import SequenceMatcher
import hashlib
import json
import random
import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from threshold import (derive_policy, pick_threshold, simulate,
                       wilson_lower_bound)  # noqa: E402

LABEL2ID = {"local_ok": 0, "escalate": 1}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def load_records(paths):
    records = []
    for p in paths:
        for line in open(p):
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


_EXPLICIT_TEMPLATE_FIELDS = (
    "template_id", "template_group", "prompt_template", "template",
)
_NUMBER_RE = re.compile(
    r"(?<!\w)(?:[$£€]\s*)?[+-]?(?:\d+(?:[.,]\d+)*)"
    r"(?:%|st|nd|rd|th)?(?!\w)", re.IGNORECASE,
)
_PROPER_NAME_RE = re.compile(
    r"\b(?:[A-Z][A-Za-z'’-]*|[A-Z]{2,})"
    r"(?:\s+(?:[A-Z][A-Za-z'’-]*|[A-Z]{2,}))*\b"
)
_TOKEN_RE = re.compile(r"<[^>]+>|[a-z0-9_]+")


def _template_scope(record):
    """Prefer the generator category over a possibly fallible classifier."""
    return str(record.get("dataset_category") or record.get("category") or "")


def canonical_prompt_template(record):
    """Return a conservative, label-free prompt-template representation.

    The generated training corpus contains parameterized math/NER prompts and
    several paraphrases of the same code or summarization task.  Exact prompt
    splitting would put those siblings on both sides of evaluation and reward
    memorization.  We first honor an explicit template id when a future dataset
    provides one, then normalize the recurring structures present today.

    This intentionally uses only prompt text/category metadata--never labels,
    correctness, verification, or model output.
    """
    scope = _template_scope(record)
    for field in _EXPLICIT_TEMPLATE_FIELDS:
        value = record.get(field)
        if value is not None and str(value).strip():
            return f"{scope}\nexplicit:{field}:{str(value).strip()}"

    text = unicodedata.normalize("NFKC", str(record.get("prompt", ""))).strip()
    category = scope.lower()

    # Code-generation siblings differ only by "Write"/"Implement"/"Create".
    if category in {"codegen", "code_generation"}:
        text = re.sub(
            r"^(?:write|implement|create)\s+(?:a\s+)?python\s+function\s+"
            r"(?:that|which)\s+", "", text, flags=re.IGNORECASE,
        )

    # Debug siblings retain the same function but wrap it in different prose.
    # Keeping the function body prevents unrelated debug tasks being collapsed.
    if category in {"debug", "code_debugging"}:
        match = re.search(r"\bdef\s+[A-Za-z_]\w*\s*\(", text)
        if match:
            text = text[match.start():]
            text = re.sub(
                r"\s+(?:find\s+and\s+fix\s+it|identify\s+the\s+bug.*)$",
                "", text, flags=re.IGNORECASE | re.DOTALL,
            )

    # The same passage is emitted with three output constraints.  Holding all
    # variants together tests transfer to new content instead of memorization.
    if category in {"summarization", "summary"}:
        text = re.sub(
            r"^summarize\s+the\s+following\s+paragraph\b.*?:\s*",
            "", text, flags=re.IGNORECASE | re.DOTALL,
        )

    text = _NUMBER_RE.sub(" <num> ", text)
    text = _PROPER_NAME_RE.sub(" <name> ", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return f"{scope}\n{text}"


def _template_similarity(left, right):
    """Similarity for normalized templates, bounded in [0, 1]."""
    if left == right:
        return 1.0
    left_tokens = set(_TOKEN_RE.findall(left))
    right_tokens = set(_TOKEN_RE.findall(right))
    union = left_tokens | right_tokens
    jaccard = len(left_tokens & right_tokens) / len(union) if union else 1.0
    sequence = SequenceMatcher(None, left, right, autojunk=False).ratio()
    # Requiring both signals avoids grouping prompts that merely share a long
    # boilerplate prefix.  The sequence-only escape hatch catches one-slot
    # factual templates such as "capital of Australia" vs "capital of Canada".
    if sequence >= 0.94:
        return sequence
    return min(sequence, jaccard)


def group_prompt_templates(records, similarity_threshold=0.82):
    """Group exact and near-duplicate prompt templates deterministically.

    Returns a list of record lists.  Grouping is conservative and scoped to the
    dataset category.  A stable sort makes membership independent of input
    order; greedy matching to a representative avoids transitive "chain"
    clusters that can swallow a whole category.
    """
    if not 0.0 <= similarity_threshold <= 1.0:
        raise ValueError("similarity_threshold must be between 0 and 1")

    explicit = defaultdict(list)
    implicit = []
    for index, record in enumerate(records):
        canonical = canonical_prompt_template(record)
        has_explicit = any(
            record.get(field) is not None and str(record.get(field)).strip()
            for field in _EXPLICIT_TEMPLATE_FIELDS
        )
        entry = (canonical, str(record.get("task_id", "")), index, record)
        if has_explicit:
            explicit[canonical].append(entry)
        else:
            implicit.append(entry)

    groups = []
    representatives = []
    for canonical in sorted(explicit):
        entries = sorted(explicit[canonical], key=lambda e: (e[1], e[2]))
        groups.append([e[3] for e in entries])
        representatives.append(canonical)

    implicit_representative_start = len(representatives)
    for canonical, _, _, record in sorted(
            implicit, key=lambda e: (e[0], e[1], e[2])):
        scope = canonical.split("\n", 1)[0]
        best_index = None
        best_similarity = -1.0
        for i in range(implicit_representative_start, len(representatives)):
            representative = representatives[i]
            if representative.split("\n", 1)[0] != scope:
                continue
            similarity = _template_similarity(canonical, representative)
            if similarity >= similarity_threshold and similarity > best_similarity:
                best_index = i
                best_similarity = similarity
        if best_index is None:
            representatives.append(canonical)
            groups.append([record])
        else:
            groups[best_index].append(record)
    return groups


def _stratum(record):
    return str(record["category"]), str(record["label"])


def grouped_train_calibration_test_split(
        records, calibration_frac=0.20, test_frac=0.20, seed=7,
        similarity_threshold=0.82, return_diagnostics=False):
    """Create grouped, approximately stratified train/calibration/test sets.

    Near-duplicate templates are indivisible groups, so exact per-stratum
    fractions are not always achievable.  A deterministic greedy objective
    balances category/label strata and total sizes while keeping every group
    wholly inside one split.
    """
    if not records:
        raise ValueError("no records to split")
    if calibration_frac <= 0 or test_frac <= 0:
        raise ValueError("calibration_frac and test_frac must be positive")
    if calibration_frac + test_frac >= 1:
        raise ValueError("calibration_frac + test_frac must be less than 1")

    groups = group_prompt_templates(records, similarity_threshold)
    split_names = ("train", "calibration", "test")
    fractions = {
        "train": 1.0 - calibration_frac - test_frac,
        "calibration": calibration_frac,
        "test": test_frac,
    }
    totals = Counter(_stratum(r) for r in records)
    target = {
        name: {key: fractions[name] * count for key, count in totals.items()}
        for name in split_names
    }
    target_size = {name: fractions[name] * len(records) for name in split_names}
    counts = {name: Counter() for name in split_names}
    sizes = Counter()
    assigned_groups = {name: [] for name in split_names}

    rng = random.Random(seed)
    decorated = []
    for group in groups:
        group_counts = Counter(_stratum(r) for r in group)
        rarity = max(1.0 / totals[key] for key in group_counts)
        stable = canonical_prompt_template(group[0])
        decorated.append((-rarity, -len(group), rng.random(), stable,
                          group, group_counts))

    def placement_delta(name, group, group_counts):
        delta = 0.0
        for key, amount in group_counts.items():
            before = counts[name][key] - target[name][key]
            after = before + amount
            delta += (after * after - before * before) / max(1, totals[key])
        before_size = sizes[name] - target_size[name]
        after_size = before_size + len(group)
        delta += 0.20 * (after_size * after_size - before_size * before_size) \
            / len(records)
        return delta

    tie_order = {name: rng.random() for name in split_names}
    for _, _, _, _, group, group_counts in sorted(decorated):
        name = min(split_names, key=lambda candidate: (
            placement_delta(candidate, group, group_counts),
            tie_order[candidate], split_names.index(candidate),
        ))
        assigned_groups[name].append(group)
        counts[name].update(group_counts)
        sizes[name] += len(group)

    # With at least three independent groups, keep all three requested splits
    # non-empty.  This only matters for tiny or unusually coarse datasets.
    for empty_name in ("calibration", "test"):
        if assigned_groups[empty_name] or len(groups) < 3:
            continue
        donors = [name for name in split_names
                  if len(assigned_groups[name]) > 1]
        if not donors:
            continue
        donor = max(donors, key=lambda name: sizes[name] - target_size[name])
        group = min(assigned_groups[donor], key=len)
        assigned_groups[donor].remove(group)
        group_counts = Counter(_stratum(r) for r in group)
        counts[donor].subtract(group_counts)
        sizes[donor] -= len(group)
        assigned_groups[empty_name].append(group)
        counts[empty_name].update(group_counts)
        sizes[empty_name] += len(group)

    split_records = {}
    for name in split_names:
        split_records[name] = [
            record for group in assigned_groups[name] for record in group
        ]
        rng.shuffle(split_records[name])

    def _group_digest(group):
        canonical = canonical_prompt_template(group[0]).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()[:12]

    diagnostics = {
        "strategy": "grouped_prompt_template_v1",
        "seed": seed,
        "fractions": fractions,
        "similarity_threshold": similarity_threshold,
        "total_template_groups": len(groups),
        "record_counts": {name: len(split_records[name]) for name in split_names},
        "template_group_counts": {
            name: len(assigned_groups[name]) for name in split_names
        },
        "strata": {
            name: {f"{cat}/{label}": count
                   for (cat, label), count in sorted(counts[name].items())
                   if count > 0}
            for name in split_names
        },
        "group_digests": {
            name: sorted(_group_digest(group) for group in assigned_groups[name])
            for name in split_names
        },
        "task_ids": {
            name: sorted(str(r.get("task_id", "")) for r in split_records[name])
            for name in split_names
        },
    }
    result = (split_records["train"], split_records["calibration"],
              split_records["test"])
    if return_diagnostics:
        return (*result, diagnostics)
    return result


def stratified_split(records, test_frac=0.25, seed=7):
    """Backward-compatible two-way wrapper with grouped template separation.

    New training code uses ``grouped_train_calibration_test_split``.  Keeping
    this helper avoids breaking notebooks that imported the old name.
    """
    train, calibration, test = grouped_train_calibration_test_split(
        records, calibration_frac=test_frac / 2, test_frac=test_frac / 2,
        seed=seed,
    )
    return train, calibration + test


def get_device(name):
    import torch
    if name in ("cuda", "rocm"):  # ROCm torch builds expose the GPU as cuda
        return torch.device("cuda")
    if name == "mps":
        return torch.device("mps")
    if name == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def metrics_from_scores(records, threshold=0.5):
    """Pure classification metrics for already-scored records."""
    tp = fp = tn = fn = 0
    for record in records:
        pred = 1 if float(record["score"]) >= threshold else 0
        true = LABEL2ID[record["label"]]
        if pred == 1 and true == 1:
            tp += 1
        elif pred == 1 and true == 0:
            fp += 1
        elif pred == 0 and true == 0:
            tn += 1
        else:
            fn += 1

    def _pr(tp_, fp_, fn_):
        precision = tp_ / (tp_ + fp_) if tp_ + fp_ else 0.0
        recall = tp_ / (tp_ + fn_) if tp_ + fn_ else 0.0
        return round(precision, 3), round(recall, 3)

    esc_p, esc_r = _pr(tp, fp, fn)
    loc_p, loc_r = _pr(tn, fn, fp)
    return {
        "threshold": threshold,
        "escalate_precision": esc_p,
        "escalate_recall": esc_r,
        "local_ok_precision": loc_p,
        "local_ok_recall": loc_r,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "accuracy": round((tp + tn) / max(1, tp + fp + tn + fn), 3),
    }


def score_all(model, tokenizer, records, device, batch_size=32, max_length=256,
              threshold=0.5):
    """Attach P(escalate) to each record, then compute classification metrics."""
    import torch
    model.eval()
    with torch.no_grad():
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            enc = tokenizer([r["prompt"] for r in batch], truncation=True,
                            padding=True, max_length=max_length,
                            return_tensors="pt").to(device)
            probs = torch.softmax(model(**enc).logits, dim=-1)[:, 1].tolist()
            for r, p in zip(batch, probs):
                r["score"] = round(float(p), 4)
    return metrics_from_scores(records, threshold)


def calibrate_policy_and_threshold(
        train_records, calibration_records, max_expected_misses=1.0,
        esc_tokens=180, esc_accuracy=0.95):
    """Derive policy from train and tune the threshold on calibration only."""
    if not train_records:
        raise ValueError("training split is empty")
    if not calibration_records:
        raise ValueError("calibration split is empty")
    if any("score" not in record for record in calibration_records):
        raise ValueError("calibration records must be scored before tuning")
    policy = derive_policy(train_records)
    threshold, projection = pick_threshold(
        calibration_records, policy,
        max_expected_misses=max_expected_misses,
        esc_tokens=esc_tokens, esc_accuracy=esc_accuracy,
    )
    return policy, threshold, projection


def summarize_training_categories(records):
    """Training-only category statistics used by the runtime configuration."""
    categories = defaultdict(list)
    for record in records:
        categories[str(record["category"])].append(record)
    summary = {}
    for category, category_records in sorted(categories.items()):
        completion_tokens = sorted(
            int(r["completion_tokens"]) for r in category_records
            if r.get("completion_tokens") is not None
        )
        p90 = completion_tokens[
            min(len(completion_tokens) - 1, int(0.9 * len(completion_tokens)))
        ] if completion_tokens else 0
        latencies = sorted(
            float(r["latency_s"]) for r in category_records
            if r.get("latency_s") is not None
        )
        p90_latency = latencies[
            min(len(latencies) - 1, int(0.9 * len(latencies)))
        ] if latencies else 0.0
        correct = sum(bool(r.get("correct")) for r in category_records)
        verified_correct = sum(
            bool(r.get("correct")) and bool(r.get("verified"))
            for r in category_records
        )
        unverified_correct = sum(
            bool(r.get("correct")) and not bool(r.get("verified"))
            for r in category_records
        )
        unverified = sum(not bool(r.get("verified"))
                         for r in category_records)
        summary[category] = {
            "n": len(category_records),
            "accuracy": round(correct / len(category_records), 3),
            "verified_correct": verified_correct,
            "unverified_correct": unverified_correct,
            "unverified_count": unverified,
            "unverified_accuracy": round(
                unverified_correct / unverified, 4) if unverified else None,
            "unverified_wilson_lower_95": round(
                wilson_lower_bound(unverified_correct, unverified), 4),
            "p90_completion_tokens": p90,
            "p90_latency_s": round(p90_latency, 3),
        }
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--stats", default=None,
                    help="optional category_stats_<tag>.json metadata from "
                         "label_local.py (routing statistics are recomputed "
                         "from train only)")
    ap.add_argument("--out", default="router_model")
    ap.add_argument("--device", default="auto",
                    choices=["auto", "cuda", "rocm", "mps", "cpu"])
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--calibration-frac", type=float, default=0.20,
                    help="fraction reserved for threshold calibration")
    ap.add_argument("--test-frac", type=float, default=0.20,
                    help="fraction reserved as untouched final test data")
    ap.add_argument("--template-similarity", type=float, default=0.82,
                    help="minimum normalized similarity for grouping likely "
                         "prompt-template siblings")
    ap.add_argument("--max-expected-misses", type=float, default=1.0,
                    help="cap on projected misses per 19 tasks when tuning "
                         "the threshold (we have ~2 spare beyond the 1 known)")
    ap.add_argument("--esc-tokens", type=float, default=180,
                    help="assumed tokens per escalation for threshold tuning; "
                         "recalibrate from eval/pick_escalation_model.py output")
    ap.add_argument("--esc-accuracy", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(argv)

    import torch
    from torch.utils.data import DataLoader
    from transformers import (DistilBertForSequenceClassification,
                              DistilBertTokenizerFast)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    records = load_records(args.labels)
    train_recs, calibration_recs, test_recs, split_diagnostics = \
        grouped_train_calibration_test_split(
            records, calibration_frac=args.calibration_frac,
            test_frac=args.test_frac, seed=args.seed,
            similarity_threshold=args.template_similarity,
            return_diagnostics=True,
        )
    if not train_recs or not calibration_recs or not test_recs:
        raise ValueError(
            "grouped split needs at least one independent template group in "
            "train, calibration, and test; add data or finer template ids"
        )
    n_esc_train = sum(1 for r in train_recs if r["label"] == "escalate")
    n_esc_calibration = sum(
        1 for r in calibration_recs if r["label"] == "escalate"
    )
    n_esc_test = sum(1 for r in test_recs if r["label"] == "escalate")
    print(f"train: {len(train_recs)} ({n_esc_train} escalate) | "
          f"calibration: {len(calibration_recs)} "
          f"({n_esc_calibration} escalate) | "
          f"test: {len(test_recs)} ({n_esc_test} escalate)")
    print("template groups: "
          f"{split_diagnostics['template_group_counts']} "
          f"({split_diagnostics['total_template_groups']} total)")

    device = get_device(args.device)
    print("device:", device)

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=2,
        id2label=ID2LABEL, label2id=LABEL2ID).to(device)

    def collate(batch):
        enc = tokenizer([r["prompt"] for r in batch], truncation=True,
                        padding=True, max_length=args.max_length,
                        return_tensors="pt")
        enc["labels"] = torch.tensor([LABEL2ID[r["label"]] for r in batch])
        return enc

    loader = DataLoader(train_recs, batch_size=args.batch_size, shuffle=True,
                        collate_fn=collate)

    n_ok_train = len(train_recs) - n_esc_train
    weight_esc = n_ok_train / max(n_esc_train, 1)
    class_weights = torch.tensor([1.0, weight_esc], dtype=torch.float32).to(device)
    print(f"class weights: local_ok=1.0, escalate={weight_esc:.2f}")
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for batch in loader:
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            loss = loss_fn(model(**batch).logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        # Calibration may guide model/epoch choices; the final test set is not
        # scored until training and threshold selection are both complete.
        metrics = score_all(model, tokenizer, calibration_recs, device,
                            max_length=args.max_length)
        print(f"epoch {epoch + 1}/{args.epochs} "
              f"loss={total_loss / max(1, len(loader)):.4f} "
              f"cal_acc={metrics['accuracy']} "
              f"esc P/R={metrics['escalate_precision']}/{metrics['escalate_recall']} "
              f"local P/R={metrics['local_ok_precision']}/{metrics['local_ok_recall']}")

    # Category policy sees train only. Threshold selection sees calibration
    # only. The test set remains untouched until both choices are frozen.
    calibration_metrics_argmax = score_all(
        model, tokenizer, calibration_recs, device,
        max_length=args.max_length,
    )
    policy, thr, calibration_projection = calibrate_policy_and_threshold(
        train_recs, calibration_recs,
        max_expected_misses=args.max_expected_misses,
        esc_tokens=args.esc_tokens, esc_accuracy=args.esc_accuracy,
    )
    calibration_metrics = metrics_from_scores(calibration_recs, thr)
    calibration_argmax_projection = simulate(
        calibration_recs, 0.5, policy, esc_tokens=args.esc_tokens,
        esc_accuracy=args.esc_accuracy,
    )
    print(f"\ntrain-derived policy: {policy}")
    print(f"chosen threshold: {thr}")
    print(f"calibration projection @19 tasks: {calibration_projection}")
    print("(calibration plain argmax 0.5: "
          f"{calibration_argmax_projection})")

    test_metrics = score_all(
        model, tokenizer, test_recs, device, max_length=args.max_length,
        threshold=thr,
    )
    test_metrics_argmax = metrics_from_scores(test_recs, 0.5)
    test_projection = simulate(
        test_recs, thr, policy, esc_tokens=args.esc_tokens,
        esc_accuracy=args.esc_accuracy,
    )
    test_argmax_projection = simulate(
        test_recs, 0.5, policy, esc_tokens=args.esc_tokens,
        esc_accuracy=args.esc_accuracy,
    )
    print(f"\nUNTOUCHED TEST metrics @ chosen threshold: {test_metrics}")
    print(f"UNTOUCHED TEST projection @19 tasks: {test_projection}")
    print(f"(test plain argmax 0.5: {test_argmax_projection})")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.cpu().save_pretrained(out)
    tokenizer.save_pretrained(out)

    training_category_stats = summarize_training_categories(train_recs)
    expected_tokens = {
        category: stats["p90_completion_tokens"]
        for category, stats in training_category_stats.items()
    }
    source_stats_metadata = {}
    if args.stats:
        raw_stats = json.loads(Path(args.stats).read_text())
        # Keep hardware/source provenance, but never feed all-data accuracy or
        # per-category summaries into runtime policy or token expectations.
        source_stats_metadata = {
            key: value for key, value in raw_stats.items()
            if key not in {"categories", "overall_accuracy"}
        }
        source_stats_metadata["path"] = args.stats
    router_config = {
        "threshold": thr,
        "label2id": LABEL2ID,
        "category_policy": policy,
        "category_policy_source": "train",
        "expected_completion_tokens": expected_tokens,
        "calibration_metrics": calibration_metrics,
        "calibration_metrics_argmax_0_5": calibration_metrics_argmax,
        "calibration_projection_19": calibration_projection,
        "calibration_projection_argmax_0_5": calibration_argmax_projection,
        "test_metrics": test_metrics,
        "test_metrics_argmax_0_5": test_metrics_argmax,
        "test_projection_19": test_projection,
        "test_projection_argmax_0_5": test_argmax_projection,
        # Compatibility for existing README/notebook consumers. These aliases
        # now unambiguously point to the untouched final test evaluation.
        "holdout_metrics": test_metrics,
        "holdout_projection_19": test_projection,
        "holdout_alias": "test",
        "evaluation_protocol": {
            "policy_fit": "train_only",
            "threshold_tuning": "calibration_only",
            "final_evaluation": "test_once_after_selection",
        },
        "split_manifest": split_diagnostics,
        "training": {"labels": args.labels, "epochs": args.epochs,
                     "seed": args.seed, "max_length": args.max_length,
                     "calibration_frac": args.calibration_frac,
                     "test_frac": args.test_frac,
                     "template_similarity": args.template_similarity,
                     "esc_tokens": args.esc_tokens,
                     "esc_accuracy": args.esc_accuracy,
                     "max_expected_misses": args.max_expected_misses},
        "category_stats": training_category_stats,
        "category_stats_source": "train",
        "source_stats_metadata": source_stats_metadata,
    }
    (out / "router_config.json").write_text(json.dumps(router_config, indent=2))
    print(f"\nsaved model + router_config.json to {out}/")


if __name__ == "__main__":
    main()
