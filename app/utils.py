"""Shared helpers: logging, atomic JSON writes, task loading, text parsing."""
import json
import logging
import os
import re
import sys
import tempfile
import time

log = logging.getLogger("agent")


def setup_logging() -> None:
    """All logs go to stderr; /output/results.json is the only artifact."""
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(handler)
    log.setLevel(logging.INFO)


def atomic_write_json(path: str, data) -> None:
    """Write JSON via temp file + rename so a crash never leaves partial JSON."""
    directory = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def load_tasks(path: str, retries: int = 5, delay: float = 1.0) -> list:
    """Read tasks.json with a small retry loop (mount may appear late)."""
    last_err = None
    for _ in range(retries):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
            last_err = ValueError("tasks.json is not a JSON array")
        except (OSError, ValueError) as e:
            last_err = e
        time.sleep(delay)
    raise RuntimeError(f"could not load {path}: {last_err}")


_NUM_RE = re.compile(r"-?\$?\d[\d,]*\.?\d*")


def extract_last_number(text: str):
    """Return the last number in text as int/float, or None."""
    matches = _NUM_RE.findall(text or "")
    if not matches:
        return None
    raw = matches[-1].replace("$", "").replace(",", "")
    # Strip a trailing period that is sentence punctuation, e.g. "195."
    if raw.endswith("."):
        raw = raw[:-1]
    if not raw or raw == "-":
        return None
    try:
        return float(raw) if "." in raw else int(raw)
    except ValueError:
        return None


def extract_labeled_line(text: str, label: str):
    """Return content after 'Label:' on the last line that carries it."""
    result = None
    pattern = re.compile(rf"^\s*\**{re.escape(label)}\**\s*[:=]\s*\**\s*(.+?)\s*$",
                         re.IGNORECASE)
    for line in (text or "").splitlines():
        m = pattern.match(line)
        if m:
            result = m.group(1).strip()
    return result


def numbers_equal(a, b, rel_tol: float = 1e-6) -> bool:
    if a is None or b is None:
        return False
    try:
        return abs(float(a) - float(b)) <= rel_tol * max(1.0, abs(float(a)), abs(float(b)))
    except (TypeError, ValueError):
        return False


def sentence_count(text: str) -> int:
    """Count sentences by terminal punctuation, ignoring common abbreviations."""
    t = re.sub(r"\b(e\.g|i\.e|etc|vs|Mr|Mrs|Ms|Dr|St)\.", r"\1", text or "")
    t = re.sub(r"\d+\.\d+", "0", t)  # decimals are not sentence ends
    parts = [p for p in re.split(r"[.!?]+(?:\s|$)", t.strip()) if p.strip()]
    return len(parts)


def word_count(text: str) -> int:
    return len(re.findall(r"\S+", text or ""))
