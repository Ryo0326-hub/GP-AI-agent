"""llama-cpp-python wrapper: load once, greedy decode, streaming with a
wall-clock budget so a slow generation can never blow the per-task limit."""
import logging
import os
import time

log = logging.getLogger("agent")


class LocalLM:
    def __init__(self, model_path: str, n_ctx: int = 4096, n_threads: int = None):
        from llama_cpp import Llama  # lazy import: tests run without the wheel
        if n_threads is None:
            n_threads = int(os.environ.get("LLM_THREADS") or min(2, os.cpu_count() or 2))
        t0 = time.monotonic()
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_batch=256,
            verbose=False,
        )
        log.info("model loaded in %.1fs (path=%s threads=%d ctx=%d)",
                 time.monotonic() - t0, model_path, n_threads, n_ctx)
        self.tok_per_sec = 8.0  # pessimistic default until benchmarked
        self.cutoff_count = 0   # generations cut off by their time budget

    def benchmark(self) -> float:
        """Generate ~30 tokens once and measure decode speed."""
        t0 = time.monotonic()
        out = self.llm.create_chat_completion(
            messages=[{"role": "user", "content": "Count from 1 to 15, comma separated."}],
            max_tokens=30, temperature=0,
        )
        dt = time.monotonic() - t0
        n = out["usage"]["completion_tokens"]
        if n > 0 and dt > 0:
            self.tok_per_sec = n / dt
        log.info("benchmark: %d tokens in %.1fs -> %.1f tok/s", n, dt, self.tok_per_sec)
        return self.tok_per_sec

    def chat(self, system: str, user: str, max_tokens: int,
             time_budget: float = 30.0) -> str:
        """Greedy streaming generation, cut off at the time budget."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        deadline = time.monotonic() + max(1.0, time_budget)
        parts = []
        stream = self.llm.create_chat_completion(
            messages=messages, max_tokens=max_tokens, temperature=0, stream=True,
        )
        for chunk in stream:
            delta = chunk["choices"][0].get("delta") or {}
            piece = delta.get("content")
            if piece:
                parts.append(piece)
            if time.monotonic() > deadline:
                self.cutoff_count += 1
                log.warning("generation cut off at time budget (%.0fs)", time_budget)
                break
        return "".join(parts).strip()
