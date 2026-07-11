#!/usr/bin/env python3
"""Dev CLI: sanity-check the trained router on a prompt or a tasks.json.

Zero Fireworks tokens — a local CPU forward pass, exactly what the runtime
does inside the container (app/router.py is the runtime twin of this file).

Usage:
  python3 router/infer_router.py "What is 15% of 240?"
  python3 router/infer_router.py --tasks train_data/tasks.json --model router_model
"""
import argparse
import json
import sys
from pathlib import Path


def load(model_dir):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
    from router import load_compact_profile, maybe_quantize
    compact = load_compact_profile(model_dir)
    if compact:
        return "compact", compact, None, compact

    import torch
    from transformers import (DistilBertForSequenceClassification,
                              DistilBertTokenizerFast)
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
    model = DistilBertForSequenceClassification.from_pretrained(model_dir)
    model = maybe_quantize(model, torch)
    model.eval()
    config = json.loads((Path(model_dir) / "router_config.json").read_text())
    return "distilbert", model, tokenizer, config


def scores(kind, model, tokenizer, prompts, batch_size=32):
    if kind == "compact":
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
        from router import score_compact
        return score_compact(prompts, model)

    import torch
    out = []
    with torch.no_grad():
        for i in range(0, len(prompts), batch_size):
            enc = tokenizer(prompts[i:i + batch_size], truncation=True,
                            padding=True, max_length=256, return_tensors="pt")
            out.extend(torch.softmax(model(**enc).logits, dim=-1)[:, 1].tolist())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt", nargs="?", help="single prompt to score")
    ap.add_argument("--tasks", help="tasks.json to score in bulk")
    ap.add_argument("--model", default="router_model")
    args = ap.parse_args()

    kind, model, tokenizer, config = load(args.model)
    thr = config["threshold"]
    if args.tasks:
        tasks = json.loads(Path(args.tasks).read_text())
        ss = scores(kind, model, tokenizer,
                    [str(t["prompt"]) for t in tasks])
        n_esc = 0
        for t, s in zip(tasks, ss):
            verdict = "escalate" if s >= thr else "local"
            n_esc += verdict == "escalate"
            print(f"{s:.3f} {verdict:8s} {t['task_id']:24s} "
                  f"{str(t['prompt'])[:70]}")
        print(f"\n{n_esc}/{len(tasks)} escalate at threshold {thr}")
    elif args.prompt:
        s = scores(kind, model, tokenizer, [args.prompt])[0]
        print(f"P(escalate)={s:.3f} threshold={thr} -> "
              f"{'escalate' if s >= thr else 'local'}")
    else:
        ap.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
