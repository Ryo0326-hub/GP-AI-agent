# AMD Hackathon Track 1 — token-efficient general-purpose agent
# linux/amd64, CPU-only, model baked at build time (never downloaded at runtime).
FROM python:3.11-slim

# Swap the model without touching code, e.g. the 1.5B if 3B is too slow on 2 vCPU:
#   --build-arg MODEL_URL=https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf
ARG MODEL_URL="https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf"

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl ca-certificates build-essential cmake \
 && rm -rf /var/lib/apt/lists/*

# Prebuilt CPU wheel index first (x86_64 only); on arm64 or any wheel miss,
# pip transparently builds llama-cpp-python from source with the toolchain above.
RUN pip install --no-cache-dir requests \
 && (pip install --no-cache-dir llama-cpp-python \
       --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu \
     || pip install --no-cache-dir llama-cpp-python)

# Model layer before code so code edits don't re-download ~2 GB.
RUN mkdir -p /models \
 && curl -fSL --retry 3 -o /models/model.gguf "$MODEL_URL" \
 && test -s /models/model.gguf

COPY app/ /app/

ENV MODEL_PATH=/models/model.gguf \
    PYTHONUNBUFFERED=1

# Writes only to /output and /tmp at runtime.
ENTRYPOINT ["python", "/app/main.py"]
