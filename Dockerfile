# AMD Hackathon Track 1 — token-efficient general-purpose agent (v2: learned router)
# linux/amd64, CPU-only, GGUF + compact router baked at build time.
FROM python:3.11.15-slim-bookworm

# The 1.5B Q4 model clears the 4 tok/s circuit breaker under the judge's 2-vCPU
# limit; the 3B measured ~2.2 tok/s and would force an all-Fireworks fallback.
ARG MODEL_URL="https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/dd26da440ef0330c47919d1ecae0966d24022222/qwen2.5-1.5b-instruct-q4_k_m.gguf"
ARG MODEL_SHA256="6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e"
# Escalation-tier preference, from eval/pick_escalation_model.py results.
# Name PATTERNS matched against the judge-injected ALLOWED_MODELS — never
# hardcoded model IDs. Empty = fall back to the size heuristics.
ARG PREFERRED_MODEL_HINTS=""
# Optional extra chat-completions params (JSON), e.g. a reasoning switch the
# escalation eval showed saves tokens. Dropped automatically on 4xx.
ARG FIREWORKS_EXTRA_BODY=""

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl ca-certificates libgomp1 build-essential cmake \
 && rm -rf /var/lib/apt/lists/*

# Prebuilt CPU wheel index first (x86_64 only); on arm64 or any wheel miss,
# pip transparently builds llama-cpp-python from source with the toolchain above.
RUN pip install --no-cache-dir "requests==2.34.2" \
 && (CMAKE_ARGS="-DGGML_NATIVE=OFF" pip install --no-cache-dir "llama-cpp-python==0.3.33" \
       --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu \
     || CMAKE_ARGS="-DGGML_NATIVE=OFF" pip install --no-cache-dir "llama-cpp-python==0.3.33")

# Keep only runtime libraries; compilers and CMake are build-time dependencies.
RUN apt-get purge -y --auto-remove build-essential cmake \
 && rm -rf /var/lib/apt/lists/*

# Model layer before code so code edits don't re-download ~1 GB.
RUN mkdir -p /models \
 && curl -fSL --retry 3 -o /models/model.gguf "$MODEL_URL" \
 && test -s /models/model.gguf \
 && if [ -n "$MODEL_SHA256" ]; then \
      echo "$MODEL_SHA256  /models/model.gguf" | sha256sum -c -; \
    fi

# The compact hashed-logistic router is a small audited JSON artifact trained
# from empirical local outcomes. It scores in milliseconds with no torch or
# transformers runtime, keeping image size, RAM, and cold start low.
RUN mkdir -p /models/router
COPY router_model/compact_router.json /models/router/compact_router.json
RUN test -s /models/router/compact_router.json
COPY app/ /app/

ENV MODEL_PATH=/models/model.gguf \
    ROUTER_DIR=/models/router \
    PREFERRED_MODEL_HINTS=$PREFERRED_MODEL_HINTS \
    FIREWORKS_EXTRA_BODY=$FIREWORKS_EXTRA_BODY \
    PYTHONUNBUFFERED=1

# Writes only to /output and /tmp at runtime.
ENTRYPOINT ["python", "/app/main.py"]
