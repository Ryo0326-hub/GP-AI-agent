#!/usr/bin/env python3
"""Dev-only: verify every model in ALLOWED_MODELS actually works.

Reads .env, sends a 5-token chat completion per model through
FIREWORKS_BASE_URL, and prints a status table. Exits non-zero if the model
the agent would select (direct-answer preference, then size heuristic) is unusable, so you can
prune .env before a real run.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "app"))

import requests  # noqa: E402


def load_env():
    path = os.path.join(ROOT, ".env")
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def probe(base, key, model):
    try:
        r = requests.post(base + "/chat/completions", timeout=30,
                          headers={"Authorization": f"Bearer {key}"},
                          json={"model": model, "temperature": 0, "max_tokens": 5,
                                "messages": [{"role": "user", "content": "Say OK"}]})
        if r.ok:
            usage = (r.json().get("usage") or {}).get("total_tokens", "?")
            return r.status_code, f"ok ({usage} tokens)"
        return r.status_code, (r.text or "").replace("\n", " ")[:160]
    except requests.RequestException as e:
        return "ERR", str(e)[:160]


def main() -> int:
    load_env()
    from fireworks import Fireworks
    fw = Fireworks()
    if not (fw.base_url and fw.api_key and fw.allowed):
        print("missing FIREWORKS_BASE_URL / FIREWORKS_API_KEY / ALLOWED_MODELS in .env")
        return 1

    print(f"base url : {fw.base_url}")
    print(f"selected : {fw.model}  (direct-answer preference, then size heuristic)\n")
    width = max(len(m) for m in fw.allowed)
    ok_models = set()
    for m in fw.allowed:
        status, msg = probe(fw.base_url, fw.api_key, m)
        mark = "OK " if str(status).startswith("2") else "BAD"
        if mark == "OK ":
            ok_models.add(m)
        print(f"  {mark} {m:<{width}}  {status}  {msg}")

    if fw.model not in ok_models:
        print(f"\nFAIL: selected model {fw.model} is unusable — prune ALLOWED_MODELS in .env")
        return 1
    print(f"\nselected model {fw.model} works; {len(ok_models)}/{len(fw.allowed)} models usable")
    return 0


if __name__ == "__main__":
    sys.exit(main())
