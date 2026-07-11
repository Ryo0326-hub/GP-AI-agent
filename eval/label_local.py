#!/usr/bin/env python3
"""Label the router-training dataset by running the FULL local pipeline
(classify -> solve_task -> deterministic verification) over every task and
grading the final answer against ground truth. Dev-only.

The label answers one question per task: "does OUR local pipeline produce a
correct final answer for this prompt?" -> local_ok / escalate. That is what
the learned compact router predicts from the prompt text alone.

Run on the host (llama-cpp-python arm64/metal is far faster than the amd64
container) or inside the dev image. Resumable: reruns skip labeled task_ids.

Usage:
  python3 eval/label_local.py --model-path ~/models/qwen2.5-3b-instruct-q4_k_m.gguf --tag 3b
  python3 eval/label_local.py --model-path ~/models/qwen2.5-1.5b-instruct-q4_k_m.gguf --tag 1.5b

Outputs:
  data/labels_<tag>.jsonl          one record per task (the training set)
  data/category_stats_<tag>.json   per-category accuracy + latency/token stats
"""
import argparse
import hashlib
import json
import os
import re
import statistics
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "app"))

from classifier import classify  # noqa: E402
from utils import extract_last_number, numbers_equal  # noqa: E402
from verify import (check_summary, extract_code, parse_summary_constraints,  # noqa: E402
                    run_code_tests)

_SENT_RE = re.compile(r"\b(positive|negative|neutral|mixed)\b", re.IGNORECASE)
_SOLVER_FILES = ("main.py", "local_model.py", "classifier.py", "prompts.py",
                 "verify.py", "utils.py")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def solver_bundle_sha256(app_dir):
    chunks = []
    for filename in sorted(_SOLVER_FILES):
        chunks.append(f"app/{filename}\0{sha256_file(os.path.join(app_dir, filename))}\n")
    return hashlib.sha256("".join(chunks).encode("utf-8")).hexdigest()


def grade(kind, expected, answer, prompt):
    """Deterministic grading, mirroring eval/judge.py where kinds overlap."""
    a = (answer or "").strip()
    if not a:
        return False
    low = a.lower()
    if kind == "number":
        return numbers_equal(extract_last_number(a), expected)
    if kind == "label":
        m = _SENT_RE.search(a)
        if not m:
            return False
        got = m.group(1).lower()
        return got == expected or (expected in ("neutral", "mixed")
                                   and got in ("neutral", "mixed"))
    if kind == "contains":
        return str(expected).lower() in low
    if kind == "contains_any":
        return any(str(e).lower() in low for e in expected)
    if kind == "contains_all":
        return all(str(e).lower() in low for e in expected)
    if kind == "entities":
        found = sum(1 for ent in expected if ent.split(":")[0].lower() in low)
        return found >= max(1, int(0.75 * len(expected)))
    if kind == "code_tests":
        code = extract_code(a)
        if not code:
            return False
        report = run_code_tests(code, expected)
        return bool(report.get("ok")) and all(r.get("passed")
                                              for r in report.get("results", []))
    if kind == "summary":
        ok, _ = check_summary(a, parse_summary_constraints(prompt))
        keyterms = expected.get("keyterms", [])
        has_content = not keyterms or any(k.lower() in low for k in keyterms)
        return ok and has_content
    raise ValueError(f"unknown grading kind: {kind}")


def load_done(path):
    done = {}
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if line:
                rec = json.loads(line)
                done[rec["task_id"]] = rec
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True, help="GGUF path")
    ap.add_argument("--tag", required=True, help="labels_<tag>.jsonl suffix, e.g. 3b")
    ap.add_argument("--tasks", default=os.path.join(ROOT, "train_data", "tasks.json"))
    ap.add_argument(
        "--source-tasks", default=None,
        help=("full canonical task manifest used for provenance when --tasks "
              "is a shard; defaults to --tasks"),
    )
    ap.add_argument("--expected", default=os.path.join(ROOT, "train_data", "expected.json"))
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "data"))
    ap.add_argument("--budget", type=float, default=25.0,
                    help="per-task local time budget (matches judge-time cap)")
    ap.add_argument("--threads", type=int, default=0,
                    help="llama threads (0 = all cores, this is a dev box)")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    if args.threads:
        os.environ["LLM_THREADS"] = str(args.threads)
    else:
        os.environ["LLM_THREADS"] = str(os.cpu_count() or 4)

    from local_model import LocalLM  # after LLM_THREADS is set
    import main as agent_main

    app_dir = os.path.dirname(os.path.abspath(agent_main.__file__))
    source_tasks = args.source_tasks or args.tasks
    provenance = {
        "schema_version": 1,
        "gguf_sha256": sha256_file(args.model_path),
        "solver_bundle_sha256": solver_bundle_sha256(app_dir),
        "labeler_sha256": sha256_file(__file__),
        "task_manifest_sha256": sha256_file(source_tasks),
        "expected_answers_sha256": sha256_file(args.expected),
        "budget_seconds": float(args.budget),
        "threads": int(os.environ["LLM_THREADS"]),
    }

    tasks = json.load(open(args.tasks))
    expected = json.load(open(args.expected))
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"labels_{args.tag}.jsonl")
    done = load_done(out_path)
    stale = [task_id for task_id, record in done.items()
             if record.get("provenance") != provenance]
    if stale:
        raise ValueError(
            "existing labels have missing or mismatched provenance "
            f"({stale[:10]}); use a fresh tag or regenerate the file")
    todo = [t for t in tasks if t["task_id"] not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(tasks)} tasks, {len(done)} already labeled, running {len(todo)}")
    if not todo:
        report(out_path, args)
        return

    lm = LocalLM(args.model_path)
    tok_s = lm.benchmark()
    print(f"model: {args.model_path} @ {tok_s:.1f} tok/s "
          f"({os.environ['LLM_THREADS']} threads)")

    with open(out_path, "a") as f:
        for i, t in enumerate(todo):
            tid, prompt = t["task_id"], str(t["prompt"])
            category = classify(prompt)
            spec = expected[tid]
            tok0 = lm.tokens_generated
            t0 = time.monotonic()
            try:
                answer, verified = agent_main.solve_task(lm, prompt, category,
                                                         args.budget)
            except Exception as e:  # noqa: BLE001
                answer, verified = "", False
                print(f"  solver error on {tid}: {e}", file=sys.stderr)
            latency = time.monotonic() - t0
            correct = grade(spec["kind"], spec["answer"], answer, prompt)
            rec = {
                "task_id": tid,
                "dataset_category": tid.rsplit("_", 1)[0],
                "category": category,
                "prompt": prompt,
                "label": "local_ok" if correct else "escalate",
                "correct": bool(correct),
                "verified": bool(verified),
                "latency_s": round(latency, 2),
                "completion_tokens": lm.tokens_generated - tok0,
                "model": args.tag,
                "provenance": provenance,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            mark = "ok " if correct else "ESC"
            print(f"[{i + 1}/{len(todo)}] {mark} {tid} ({category}) "
                  f"{latency:.1f}s {rec['completion_tokens']}tok "
                  f"verified={verified}")
    report(out_path, args, tok_s=lm.tok_per_sec)


def report(out_path, args, tok_s=None):
    """Per-category stats: the headline table that drives routing policy."""
    recs = list(load_done(out_path).values())
    if not recs:
        print("no labels yet")
        return
    cats = sorted({r["category"] for r in recs})
    stats = {}
    print(f"\n=== local pipeline accuracy per category ({args.tag}) ===")
    print(f"{'category':15s} {'n':>4s} {'correct':>8s} {'acc':>6s} "
          f"{'ver+ok':>7s} {'unver+ok':>9s} {'p90 tok':>8s} {'mean s':>7s}")
    for cat in cats:
        rs = [r for r in recs if r["category"] == cat]
        n = len(rs)
        ok = sum(r["correct"] for r in rs)
        ver_ok = sum(r["correct"] and r["verified"] for r in rs)
        unver_ok = sum(r["correct"] and not r["verified"] for r in rs)
        toks = sorted(r.get("completion_tokens", 0) for r in rs)
        p90_tok = toks[min(len(toks) - 1, int(0.9 * len(toks)))] if toks else 0
        latencies = sorted(float(r["latency_s"]) for r in rs)
        p90_latency = latencies[
            min(len(latencies) - 1, int(0.9 * len(latencies)))]
        mean_s = statistics.mean(r["latency_s"] for r in rs)
        acc = ok / n
        stats[cat] = {"n": n, "accuracy": round(acc, 3),
                      "verified_correct": ver_ok,
                      "unverified_correct": unver_ok,
                      "p90_completion_tokens": p90_tok,
                      "p90_latency_s": round(p90_latency, 2),
                      "mean_latency_s": round(mean_s, 2)}
        print(f"{cat:15s} {n:4d} {ok:8d} {acc:6.0%} {ver_ok:7d} {unver_ok:9d} "
              f"{p90_tok:8d} {mean_s:7.1f}")
    total = len(recs)
    total_ok = sum(r["correct"] for r in recs)
    print(f"{'overall':15s} {total:4d} {total_ok:8d} {total_ok / total:6.0%}")
    stats_path = os.path.join(args.out_dir, f"category_stats_{args.tag}.json")
    if tok_s is None and os.path.exists(stats_path):
        try:
            tok_s = json.load(open(stats_path)).get("tok_s_devbox")
        except (OSError, ValueError, AttributeError):
            pass
    meta = {"tag": args.tag, "tok_s_devbox": tok_s, "categories": stats,
            "overall_accuracy": round(total_ok / total, 3)}
    with open(stats_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nwrote {stats_path}")
    print("policy guidance: <70% overall -> always escalate; trust unverified "
          "only when its 95% Wilson lower bound is >=90%")


if __name__ == "__main__":
    main()
