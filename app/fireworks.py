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
import json
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
# Build-time extras are deliberately narrow. In particular, they can never
# replace the runtime allowlisted model, messages, token cap, or headers.
_SAFE_EXTRA_BODY_KEYS = frozenset({"reasoning_effort"})


def _preferred_hints():
    """Name-pattern preference order, baked at build time from the empirical
    eval/pick_escalation_model.py results (patterns, never full model IDs)."""
    raw = os.environ.get("PREFERRED_MODEL_HINTS") or ""
    return [h.strip().lower() for h in raw.split(",") if h.strip()]


def _retry_hints():
    """Name patterns for the verification-retry tier (a different, usually
    stronger allowed model that re-attempts an answer that failed the
    deterministic checks)."""
    raw = os.environ.get("RETRY_MODEL_HINTS") or ""
    return [h.strip().lower() for h in raw.split(",") if h.strip()]


def _extra_body():
    """Optional extra request params (e.g. reasoning controls) baked from the
    escalation-model eval. Malformed or unsupported fields are ignored."""
    raw = os.environ.get("FIREWORKS_EXTRA_BODY") or ""
    if not raw:
        return {}
    try:
        extra = json.loads(raw)
    except ValueError:
        log.warning("ignoring malformed FIREWORKS_EXTRA_BODY")
        return {}
    if not isinstance(extra, dict):
        log.warning("ignoring non-object FIREWORKS_EXTRA_BODY")
        return {}
    ignored = sorted(set(extra) - _SAFE_EXTRA_BODY_KEYS)
    if ignored:
        # Log keys only: a mistakenly supplied credential must never be echoed.
        log.warning("ignoring unsupported FIREWORKS_EXTRA_BODY keys: %s", ignored)
    return {key: extra[key] for key in _SAFE_EXTRA_BODY_KEYS if key in extra}

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
        self.derived_usage_calls = 0  # total reconstructed from prompt+completion
        self.unknown_usage_calls = 0  # successful calls with no usable token total
        self.definitive_unavailable = False
        self.unsupported_extra_models = set()
        self._lock = threading.Lock()
        self.model = self._pick_model()

    @property
    def available(self) -> bool:
        return bool(self.base_url and self.api_key and self.model
                    and not self.definitive_unavailable)

    def _pick_model(self):
        """Empirically-preferred name patterns first (PREFERRED_MODEL_HINTS,
        from eval/pick_escalation_model.py), then the size heuristics, then
        the first usable model."""
        candidates = [m for m in self.allowed if m not in self.bad_models]
        if not candidates:
            return None
        for hint in _preferred_hints():
            for m in candidates:
                if hint in m.lower():
                    return m
        # Prefer a capable model whose API separates reasoning from final content;
        # this avoids truncated chain-of-thought consuming the answer budget.
        for pattern in (_DIRECT_ANSWER_HINTS, _SMALL_HINTS, _MID_HINTS):
            for m in candidates:
                if pattern.search(m):
                    return m
        return candidates[0]

    def pick_distinct(self, exclude, hints):
        """First usable allowed model not in ``exclude``, preferring the given
        name patterns, then larger-looking siblings."""
        with self._lock:
            candidates = [m for m in self.allowed
                          if m not in self.bad_models and m not in exclude]
        if not candidates:
            return None
        for hint in list(hints) + ["pro", "k2", "large", "70b", "405b"]:
            for m in candidates:
                if hint in m.lower():
                    return m
        return candidates[0]

    def secondary_model(self):
        """A different allowed model for one verification-driven retry.

        RETRY_MODEL_HINTS patterns win; otherwise prefer a larger-looking
        sibling. Returns None when no distinct usable model exists, in which
        case the caller skips the retry (temperature-0 replays are pointless).
        """
        return self.pick_distinct({self.model}, _retry_hints())

    def _mark_bad(self, model):
        with self._lock:
            self.bad_models.add(model)
            self.model = self._pick_model()
        if self.model:
            log.error("model %s marked unusable; switching to %s", model, self.model)
        else:
            log.error("model %s marked unusable; no allowed models left", model)

    def complete(self, prompt: str, max_tokens: int, timeout: float = 25.0,
                 system: str = None, model_override: str = None) -> str:
        """One escalation within one overall timeout. Returns '' on failure.

        A caller may supply the empirically selected category system prompt;
        per-category max_tokens caps still bound the completion.
        ``model_override`` must already be an ALLOWED_MODELS member (the
        retry tier passes ``secondary_model()``); a 404 on it falls back to
        the primary selection on the next loop iteration.
        """
        deadline = time.monotonic() + max(1.0, timeout)
        retried = False
        configured_extra = _extra_body()
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                log.warning("escalation exceeded overall %.1fs timeout", timeout)
                return ""
            model = self.model
            if (model_override and model_override in self.allowed
                    and model_override not in self.bad_models):
                model = model_override
            if not self.available or model is None:
                return ""
            with self._lock:
                extra = ({} if model in self.unsupported_extra_models
                         else dict(configured_extra))
            with self._lock:
                self.calls_attempted += 1
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            body = dict(extra)
            # Mandatory values are written last as defense in depth, even though
            # _extra_body already rejects every reserved/unknown key.
            body.update({
                "model": model,
                "messages": messages,
                "temperature": 0,
                "max_tokens": max_tokens,
            })
            headers = {"Authorization": f"Bearer {self.api_key}",
                       "Content-Type": "application/json"}
            # requests has no strict whole-operation wall-clock timeout. A
            # bounded connect/read tuple plus the outer deadline is the
            # strongest in-process bound it provides; every retry recomputes
            # these phase budgets from the remaining allowance.
            connect_timeout = min(3.0, max(0.25, remaining * 0.20))
            read_timeout = max(0.25, remaining - connect_timeout - 0.25)
            try:
                resp = requests.post(self.base_url + "/chat/completions",
                                     json=body, headers=headers,
                                     timeout=(connect_timeout, read_timeout))
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
                tokens, usage_source = _usage_tokens(data.get("usage"))
                with self._lock:
                    if tokens is not None:
                        self.total_tokens += tokens
                    if usage_source == "derived":
                        self.derived_usage_calls += 1
                    elif usage_source == "unknown":
                        self.unknown_usage_calls += 1
                if text:
                    with self._lock:
                        self.calls_succeeded += 1
                    log.info("escalation ok: model=%s tokens=%s usage=%s "
                             "(known running total=%d)",
                             model, tokens if tokens is not None else "unknown",
                             usage_source, self.total_tokens)
                    return text
                log.warning("escalation returned empty content (model=%s)", model)
                if retried or deadline - time.monotonic() <= 1.0:
                    return ""
                retried = True
                continue

            snippet = (resp.text or "")[:_BODY_SNIPPET]
            if resp.status_code in (401, 403):
                with self._lock:
                    self.definitive_unavailable = True
                log.error("escalation HTTP %d (definitive auth failure): %s",
                          resp.status_code, snippet)
                return ""
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
            if extra:
                # A baked extra param (e.g. a reasoning switch) may not be
                # supported by this endpoint. Cache that fact per model so
                # later tasks do not each pay for the same rejected request.
                with self._lock:
                    self.unsupported_extra_models.add(model)
                log.warning("retrying once without FIREWORKS_EXTRA_BODY params")
                continue
            return ""


def _usage_tokens(raw_usage):
    """Return ``(tokens, source)`` without invalidating a usable answer.

    Fireworks normally returns ``total_tokens``. If it is absent, prompt plus
    completion usage is an exact reconstruction under the OpenAI-compatible
    schema. Missing or malformed usage remains explicitly unknown rather than
    being reported as zero.
    """
    usage = raw_usage if isinstance(raw_usage, dict) else {}

    def nonnegative_int(value):
        if isinstance(value, bool):
            return None
        try:
            parsed = int(value)
        except (TypeError, ValueError, OverflowError):
            return None
        return parsed if parsed >= 0 else None

    total = nonnegative_int(usage.get("total_tokens"))
    if total is not None:
        return total, "provider"
    prompt_tokens = nonnegative_int(usage.get("prompt_tokens"))
    completion_tokens = nonnegative_int(usage.get("completion_tokens"))
    if prompt_tokens is not None and completion_tokens is not None:
        return prompt_tokens + completion_tokens, "derived"
    return None, "unknown"
