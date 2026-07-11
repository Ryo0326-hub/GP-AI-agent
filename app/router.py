"""Runtime router: score every task's P(escalate) locally at startup.

The preferred model is a tiny hashed-logistic JSON profile.  It has no model
runtime dependency, uses the same features as the browser demo, and loads in
milliseconds.  Existing DistilBERT checkpoints remain a transparent fallback
and are freed before the GGUF loads so the two large models never share peak
RAM.

Zero Fireworks tokens — a local CPU forward pass. Degrades gracefully: if the
model directory, torch, or transformers are missing or broken, scoring
returns None and the orchestrator falls back to verification-gated local
solving for everything (v1 behavior), so a router problem can never take the
agent down at judge time.
"""
import gc
import json
import logging
import math
import os
import re
import time

log = logging.getLogger("agent")

ROUTER_DIR = os.environ.get("ROUTER_DIR", "/models/router")
COMPACT_PROFILE_NAME = "compact_router.json"

_TOKEN_RE = re.compile(r"[a-z0-9_]+|[^\sA-Za-z0-9_]")
_CODE_HINT_RE = re.compile(
    r"```|\bdef \w+|\bfunction\b|\breturn\b|\bclass \w+\(|=>|"
    r"\bprintln\b|\bconsole\.log\b",
    re.IGNORECASE | re.ASCII,
)


def _js_unicode_scalars(value):
    """Normalize Python strings like JS Unicode iteration + TextEncoder.

    ``JSON.parse`` can leave UTF-16 surrogate pairs visible to Python's JSON
    decoder while JavaScript's ``/u`` regex and ``Array.from`` see one scalar.
    This round-trip combines valid pairs and maps lone surrogates to U+FFFD,
    which is also what ``TextEncoder`` emits.
    """
    return str(value).encode("utf-16-le", "surrogatepass").decode(
        "utf-16-le", "replace")


def fnv1a_32(value):
    """Return the unsigned FNV-1a hash of a UTF-8 string.

    This deliberately mirrors ``Math.imul`` + ``TextEncoder`` in the browser
    router rather than Python's randomized built-in hash.
    """
    hashed = 0x811C9DC5
    for byte in _js_unicode_scalars(value).encode("utf-8"):
        hashed ^= byte
        hashed = (hashed * 0x01000193) & 0xFFFFFFFF
    return hashed


def compact_feature_counts(prompt, category, dimension):
    """Return the browser-compatible sparse hashed feature counts."""
    if not isinstance(dimension, int) or isinstance(dimension, bool) \
            or dimension <= 0:
        raise ValueError("compact router dimension must be a positive integer")
    text = _js_unicode_scalars(prompt or "")
    tokens = _TOKEN_RE.findall(text.lower())[:256]
    raw = [
        f"category:{category}",
        f"length:{min(8, len(text) // 80)}",
    ]
    for index, token in enumerate(tokens):
        raw.append(f"u:{token}")
        if index:
            raw.append(f"b:{tokens[index - 1]}|{token}")
    if re.search(r"[0-9]", text):
        raw.append("shape:digits")
    if _CODE_HINT_RE.search(text):
        raw.append("shape:code")
    if "?" in text:
        raw.append("shape:question")

    counts = {}
    for feature in raw:
        feature_index = fnv1a_32(feature) % dimension
        counts[feature_index] = counts.get(feature_index, 0) + 1
    return counts


def compact_feature_values(prompt, category, dimension):
    """Return transformed values consumed by hashed-logistic profiles."""
    return {
        index: 1.0 + math.log(count)
        for index, count in compact_feature_counts(
            prompt, category, dimension).items()
    }


def _finite_number(value, field):
    if isinstance(value, bool) or not isinstance(value, (int, float)) \
            or not math.isfinite(float(value)):
        raise ValueError(f"compact router {field} must be a finite number")
    return float(value)


def validate_compact_profile(profile):
    """Validate and normalize an untrusted compact-router JSON object."""
    if not isinstance(profile, dict):
        raise ValueError("compact router profile must be a JSON object")
    if profile.get("modelType") != "hashed-logistic-v1":
        raise ValueError("unsupported compact router modelType")
    if profile.get("schemaVersion") != 1:
        raise ValueError("unsupported compact router schemaVersion")
    dimension = profile.get("dimension")
    if isinstance(dimension, bool) or not isinstance(dimension, int) \
            or not 0 < dimension <= 1_000_000:
        raise ValueError("compact router dimension is invalid")
    bias = _finite_number(profile.get("bias"), "bias")
    threshold = _finite_number(profile.get("threshold"), "threshold")
    # router.threshold uses 1.01 as its deterministic "route none" sentinel.
    if not 0.0 <= threshold <= 1.01:
        raise ValueError("compact router threshold is outside [0, 1.01]")
    raw_weights = profile.get("weights")
    if not isinstance(raw_weights, dict):
        raise ValueError("compact router weights must be an object")
    weights = {}
    for raw_index, raw_weight in raw_weights.items():
        try:
            index = int(raw_index)
        except (TypeError, ValueError) as exc:
            raise ValueError("compact router weight index is invalid") from exc
        if str(index) != str(raw_index) or not 0 <= index < dimension:
            raise ValueError("compact router weight index is out of range")
        weights[index] = _finite_number(raw_weight, f"weight[{index}]")
    normalized = dict(profile)
    normalized.update({
        "dimension": dimension,
        "bias": bias,
        "threshold": threshold,
        "weights": weights,
    })
    if "categoryPolicy" in profile:
        raw_policy = profile["categoryPolicy"]
        if not isinstance(raw_policy, dict):
            raise ValueError("compact router categoryPolicy must be an object")
        policy = {}
        for key in ("always_escalate", "trust_local"):
            values = raw_policy.get(key, [])
            if not isinstance(values, list) or not all(
                    isinstance(value, str) for value in values):
                raise ValueError(
                    f"compact router categoryPolicy.{key} must be a string array")
            policy[key] = list(values)
        normalized["categoryPolicy"] = policy
    if "expectedCompletionTokens" in profile:
        raw_expected = profile["expectedCompletionTokens"]
        if not isinstance(raw_expected, dict):
            raise ValueError(
                "compact router expectedCompletionTokens must be an object")
        expected = {}
        for category, raw_value in raw_expected.items():
            if not isinstance(category, str):
                raise ValueError(
                    "compact router expected token category is invalid")
            value = _finite_number(
                raw_value, f"expectedCompletionTokens[{category}]")
            if value < 0:
                raise ValueError(
                    "compact router expected completion tokens must be non-negative")
            expected[category] = value
        normalized["expectedCompletionTokens"] = expected
    if "expectedLocalLatencySeconds" in profile:
        raw_latency = profile["expectedLocalLatencySeconds"]
        if not isinstance(raw_latency, dict):
            raise ValueError(
                "compact router expectedLocalLatencySeconds must be an object")
        latency = {}
        for category, raw_value in raw_latency.items():
            if not isinstance(category, str):
                raise ValueError(
                    "compact router expected latency category is invalid")
            value = _finite_number(
                raw_value, f"expectedLocalLatencySeconds[{category}]")
            if value < 0:
                raise ValueError(
                    "compact router expected local latency must be non-negative")
            latency[category] = value
        normalized["expectedLocalLatencySeconds"] = latency
    return normalized


def _compact_profile_candidates(router_dir):
    explicit = os.environ.get("COMPACT_ROUTER_PROFILE")
    if explicit:
        yield explicit if os.path.isabs(explicit) \
            else os.path.join(router_dir, explicit)
    yield os.path.join(router_dir, COMPACT_PROFILE_NAME)


def load_compact_profile(router_dir=None, warn=False):
    """Load the first valid compact profile, or ``None`` when absent/broken."""
    d = router_dir or ROUTER_DIR
    seen = set()
    for path in _compact_profile_candidates(d):
        if path in seen:
            continue
        seen.add(path)
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                return validate_compact_profile(json.load(f))
        except (OSError, ValueError, TypeError) as exc:
            log.warning("invalid compact router profile (%s): %s", path, exc)
    if warn:
        log.warning("compact router profile missing under %s", d)
    return None


def _classify_for_compact(prompt):
    # Lazy import keeps this module independently testable and works both when
    # /app is on sys.path in the image and when tests import app.router.
    try:
        from classifier import classify
    except ImportError:  # pragma: no cover - package-style import convenience
        from app.classifier import classify
    return classify(str(prompt or ""))


def _score_validated_compact_prompt(prompt, profile, category=None):
    category = category or _classify_for_compact(prompt)
    logit = float(profile["bias"])
    for index, value in compact_feature_values(
            prompt, category, int(profile["dimension"])).items():
        logit += float(profile["weights"].get(index, 0.0)) * value
    bounded = max(-30.0, min(30.0, logit))
    return 1.0 / (1.0 + math.exp(-bounded))


def score_compact_prompt(prompt, profile, category=None):
    """Score one prompt after validating its compact profile."""
    return _score_validated_compact_prompt(
        prompt, validate_compact_profile(profile), category)


def score_compact(prompts, profile):
    """Score prompts in order with the zero-dependency compact router."""
    validated = validate_compact_profile(profile)
    return [_score_validated_compact_prompt(prompt, validated)
            for prompt in prompts]


def load_config(router_dir=None):
    """Load routing policy and the threshold for the preferred scorer.

    A compact profile's threshold overrides a legacy DistilBERT config because
    the threshold is calibrated against that profile's own score distribution.
    """
    d = router_dir or ROUTER_DIR
    path = os.path.join(d, "router_config.json")
    config = None
    try:
        with open(path, "r", encoding="utf-8") as f:
            value = json.load(f)
            if isinstance(value, dict):
                config = value
            else:
                log.warning("router config is not an object: %s", path)
    except (OSError, ValueError) as e:
        log.warning("no router config (%s): %s", path, e)
    profile = load_compact_profile(d)
    if profile:
        config = dict(config or {})
        config["threshold"] = profile["threshold"]
        config["router_model_type"] = profile["modelType"]
        config["router_revision"] = profile.get("revision", "unknown")
        if "categoryPolicy" in profile:
            config["category_policy"] = profile["categoryPolicy"]
            config["category_policy_source"] = "compact_profile_train"
        if "expectedCompletionTokens" in profile:
            expected = profile["expectedCompletionTokens"]
            latency = profile.get("expectedLocalLatencySeconds") or {}
            if latency:
                # Keep the legacy config key consumed by app/main.py, but give
                # budget.py both measured end-to-end latency and completion
                # tokens.  Older profiles continue to expose plain numbers.
                config["expected_completion_tokens"] = {
                    category: {
                        "completion_tokens": tokens,
                        "p90_latency_s": latency.get(category, 0.0),
                    }
                    for category, tokens in expected.items()
                }
            else:
                config["expected_completion_tokens"] = expected
            config["expected_completion_tokens_source"] = \
                "compact_profile_train"
    return config


def score_and_free(prompts, router_dir=None, max_length=256, batch_size=16):
    """Return P(escalate) per prompt, or None if the router can't run.

    Prefer the compact zero-dependency profile.  If absent or invalid, load
    DistilBERT, apply int8 dynamic quantization, score in batches, then release
    the weights — peak RAM stays a ~70 MB blip before llama.cpp allocates.
    """
    d = router_dir or ROUTER_DIR
    t0 = time.monotonic()
    profile = load_compact_profile(d)
    if profile:
        try:
            scores = score_compact(prompts, profile)
        except Exception as e:  # noqa: BLE001 - fallback must remain available
            log.error("compact router scoring failed; trying DistilBERT: %s", e)
        else:
            log.info("compact router scored %d tasks in %.3fs (revision=%s)",
                     len(prompts), time.monotonic() - t0,
                     profile.get("revision", "unknown"))
            return scores

    if not os.path.exists(os.path.join(d, "config.json")):
        log.warning("router model missing at %s; routing disabled", d)
        return None
    t0 = time.monotonic()
    try:
        scores = _score(prompts, d, max_length, batch_size)
    except Exception as e:  # noqa: BLE001 — any router failure must be survivable
        log.error("router scoring failed, falling back to no-router mode: %s", e)
        return None
    finally:
        gc.collect()  # _score's model/tokenizer are out of scope here
    log.info("router scored %d tasks in %.1fs", len(prompts),
             time.monotonic() - t0)
    return scores


def maybe_quantize(model, torch):
    """int8 dynamic quantization (~4x smaller, faster on 2 vCPU) where the
    torch build has a quantization engine; fp32 otherwise (e.g. some arm64
    builds ship NoQEngine)."""
    try:
        engines = getattr(torch.backends.quantized, "supported_engines", [])
        if all(e == "none" for e in engines):
            log.info("no torch quantization engine; router stays fp32")
            return model
        return torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8)
    except Exception as e:  # noqa: BLE001
        log.warning("router quantization unavailable, staying fp32: %s", e)
        return model


def _score(prompts, model_dir, max_length, batch_size):
    import torch
    from transformers import (DistilBertForSequenceClassification,
                              DistilBertTokenizerFast)
    torch.set_num_threads(int(os.environ.get("LLM_THREADS") or 2))
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
    model = DistilBertForSequenceClassification.from_pretrained(model_dir)
    model = maybe_quantize(model, torch)
    model.eval()
    scores = []
    with torch.no_grad():
        for i in range(0, len(prompts), batch_size):
            enc = tokenizer(prompts[i:i + batch_size], truncation=True,
                            padding=True, max_length=max_length,
                            return_tensors="pt")
            probs = torch.softmax(model(**enc).logits, dim=-1)[:, 1]
            scores.extend(float(p) for p in probs)
    return scores
