"""Fireworks escalation client.

All calls go through FIREWORKS_BASE_URL (OpenAI-compatible /chat/completions)
with FIREWORKS_API_KEY, using only models listed in ALLOWED_MODELS. All three
come from the environment at runtime — never hardcoded.

Error policy:
- 404 is non-retryable: the model is marked known-bad for the rest of the run
  and the next-smallest allowed model is selected instead.
- 408/429/5xx and network errors get one retry with backoff.
- Other 4xx are non-retryable failures.
Response bodies are logged (truncated) so a wrong model ID is distinguishable
from a wrong route. Thread-safe: bulk escalation calls this concurrently.
"""
import logging
import os
import re
import threading
import time
import unicodedata

import requests

log = logging.getLogger("agent")

_SMALL_HINTS = re.compile(r"(^|[^0-9])(0\.5|1|1\.5|2|3|4)b\b|mini|small|tiny|nano|flash|lite", re.IGNORECASE)
_MID_HINTS = re.compile(r"(^|[^0-9])(7|8|9|11|12|13|14)b\b", re.IGNORECASE)
_DIRECT_ANSWER_HINTS = re.compile(r"gpt[-_]?oss", re.IGNORECASE)

_RETRYABLE_STATUS = {408, 429}
_BODY_SNIPPET = 300


class Fireworks:
    def __init__(self):
        self.base_url = (os.environ.get("FIREWORKS_BASE_URL") or "").rstrip("/")
        self.api_key = os.environ.get("FIREWORKS_API_KEY") or ""
        raw = os.environ.get("ALLOWED_MODELS") or ""
        self.allowed = [m.strip() for m in raw.split(",") if m.strip()]
        self.bad_models = set()      # 404'd this run; never re-tried
        self.total_tokens = 0        # prompt+completion, from API usage
        self.calls_attempted = 0
        self.calls_succeeded = 0
        self._lock = threading.Lock()
        self.model = self._pick_model()

    @property
    def available(self) -> bool:
        return bool(self.base_url and self.api_key and self.model)

    def _pick_model(self):
        """Smallest usable model by name heuristics; fall back to first usable."""
        candidates = [m for m in self.allowed if m not in self.bad_models]
        if not candidates:
            return None
        # Prefer a capable model whose API separates reasoning from final content;
        # this avoids truncated chain-of-thought consuming the answer budget.
        for pattern in (_DIRECT_ANSWER_HINTS, _SMALL_HINTS, _MID_HINTS):
            for m in candidates:
                if pattern.search(m):
                    return m
        return candidates[0]

    def _mark_bad(self, model):
        with self._lock:
            self.bad_models.add(model)
            self.model = self._pick_model()
        if self.model:
            log.error("model %s marked unusable; switching to %s", model, self.model)
        else:
            log.error("model %s marked unusable; no allowed models left", model)

    def complete(self, prompt: str, max_tokens: int, timeout: float = 25.0) -> str:
        """One escalation within one overall timeout. Returns '' on failure."""
        deadline = time.monotonic() + max(1.0, timeout)
        retried = False
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                log.warning("escalation exceeded overall %.1fs timeout", timeout)
                return ""
            model = self.model
            if not self.available or model is None:
                return ""
            with self._lock:
                self.calls_attempted += 1
            body = {
                "model": model,
                "messages": [
                    {"role": "system", "content": (
                        "Return only the final answer, concise but complete. Do not narrate "
                        "your planning or hidden reasoning. For programming tasks, begin "
                        "immediately with the requested code block."
                    )},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0,
                "max_tokens": max_tokens,
            }
            headers = {"Authorization": f"Bearer {self.api_key}",
                       "Content-Type": "application/json"}
            try:
                resp = requests.post(self.base_url + "/chat/completions",
                                     json=body, headers=headers, timeout=remaining)
            except requests.RequestException as e:
                log.warning("escalation network error (model=%s): %s", model, e)
                if retried:
                    return ""
                retried = True
                if deadline - time.monotonic() <= 1.0:
                    return ""
                time.sleep(1.0)
                continue

            if resp.ok:
                try:
                    data = resp.json()
                    text = (data["choices"][0]["message"].get("content") or "").strip()
                    text = unicodedata.normalize("NFKC", text)
                except (ValueError, KeyError, IndexError, TypeError) as e:
                    log.warning("escalation unparseable 2xx body (model=%s): %s", model, e)
                    return ""
                usage = data.get("usage") or {}
                tokens = int(usage.get("total_tokens") or 0)
                with self._lock:
                    self.total_tokens += tokens
                if text:
                    with self._lock:
                        self.calls_succeeded += 1
                    log.info("escalation ok: model=%s tokens=%d (running total=%d)",
                             model, tokens, self.total_tokens)
                    return text
                log.warning("escalation returned empty content (model=%s)", model)
                if retried or deadline - time.monotonic() <= 1.0:
                    return ""
                retried = True
                continue

            snippet = (resp.text or "")[:_BODY_SNIPPET]
            if resp.status_code == 404:
                log.error("escalation HTTP 404 (non-retryable, model=%s): %s",
                          model, snippet)
                self._mark_bad(model)
                continue  # immediately try the next-smallest allowed model
            if resp.status_code in _RETRYABLE_STATUS or resp.status_code >= 500:
                log.warning("escalation HTTP %d (retryable, model=%s): %s",
                            resp.status_code, model, snippet)
                if retried:
                    return ""
                retried = True
                if deadline - time.monotonic() <= 1.0:
                    return ""
                time.sleep(1.0)
                continue
            log.error("escalation HTTP %d (non-retryable, model=%s): %s",
                      resp.status_code, model, snippet)
            return ""
