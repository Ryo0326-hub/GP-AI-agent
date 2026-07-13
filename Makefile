IMAGE ?= rkitano/gp-agent:v12
DEV_IMAGE ?= gp-agent:v12-dev-arm
PY ?= .venv/bin/python
ROUTER_TAG ?= 15b_2cpu_v11
ROUTER_LABELS ?= data/labels_$(ROUTER_TAG).jsonl
ROUTER_TASKS ?= train_data/tasks.json
ROUTER_CORE ?= router_model/compact_router.json
ROUTER_DEMO ?= demo/src/data/router-profile.json
MODEL_3B ?= models/qwen2.5-3b-instruct-q4_k_m.gguf
MODEL_15B ?= models/qwen2.5-1.5b-instruct-q4_k_m.gguf
MODEL_3B_URL = https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf
MODEL_15B_URL = https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/dd26da440ef0330c47919d1ecae0966d24022222/qwen2.5-1.5b-instruct-q4_k_m.gguf
# Measured on the 20-task hard set: category prompt + reasoning-off Flash was
# 19/20 at 3,365 tokens, beating the raw prompt by +10 accuracy points / -24.6%.
# Ordered measured fallback: Flash first (19/20), then Pro (18/20) before the
# generic name heuristics can select the weaker GPT-OSS baseline.
HINTS ?= deepseek-v4-flash,deepseek-v4-pro
# Verification-retry tier: Pro was the second-best category-prompt scorer
# (18/20); it re-attempts only answers that fail the deterministic checks.
RETRY_HINTS ?= deepseek-v4-pro,kimi
EXTRA_BODY ?= {"reasoning_effort":"none"}
BUILD_ARGS = --build-arg PREFERRED_MODEL_HINTS="$(HINTS)" --build-arg FIREWORKS_EXTRA_BODY='$(EXTRA_BODY)' --build-arg RETRY_MODEL_HINTS="$(RETRY_HINTS)"
# v12 changed only remote orchestration in app/main.py (remote-first plan,
# hosted-answer verification, factual consensus); the local solvers the labels
# measure are byte-identical, which the checker proves by reconstructing the
# recorded aggregate digest from the unwaived files. Drop the waiver after the
# next relabel run.
ROUTER_DRIFT_FLAGS ?= --allow-solver-drift app/main.py
ROUTER_ARTIFACT_CHECK = python3 eval/check_router_artifacts.py \
	--core $(ROUTER_CORE) --demo $(ROUTER_DEMO) \
	--labels $(ROUTER_LABELS) --require-tasks $(ROUTER_TASKS) \
	$(ROUTER_DRIFT_FLAGS)

.PHONY: build build-dev build-3b build-15b build-remote build-push push test testset dataset \
		venv models-download label-3b label-15b train-router train-distilbert router-check \
		router-artifact-check pick-model \
		run-local rehearsal accuracy judge judge-19 check-models all

build: router-artifact-check
	docker buildx build --platform linux/amd64 $(BUILD_ARGS) -t $(IMAGE) --load .

# Native Apple Silicon dev build — no qemu emulation, real tok/s for iteration.
build-dev: router-artifact-check
	docker buildx build --platform linux/arm64 $(BUILD_ARGS) -t $(DEV_IMAGE) --load .

# Default/final model: the 1.5B GGUF is already the Dockerfile default.
build-15b: router-artifact-check
	docker buildx build --platform linux/amd64 $(BUILD_ARGS) \
	  --build-arg MODEL_URL=$(MODEL_15B_URL) -t $(IMAGE) --load .

# Accuracy experiment only: the 3B measured below the judge-time speed floor.
build-3b: router-artifact-check
	docker buildx build --platform linux/amd64 $(BUILD_ARGS) \
	  --build-arg MODEL_URL=$(MODEL_3B_URL) --build-arg MODEL_SHA256= \
	  -t $(IMAGE) --load .

# Lightweight accuracy-first fallback when publishing the bundled GGUF is slow.
build-remote:
	docker buildx build --platform linux/amd64 -f Dockerfile.remote -t $(IMAGE) --load .

# Final submission build+push (always amd64).
build-push: router-artifact-check
	docker buildx build --platform linux/amd64 $(BUILD_ARGS) -t $(IMAGE) --push .

push: build-push

# ---- v2 router pipeline (dev machine / AMD box) ----

venv:
	uv venv .venv --python 3.13
	uv pip install -p .venv/bin/python cmake ninja   # llama-cpp source build needs them
	PATH="$(PWD)/.venv/bin:$$PATH" uv pip install -p .venv/bin/python -r eval/requirements-dev.txt

models-download:
	mkdir -p models
	[ -f $(MODEL_3B) ] || curl -fSL --retry 3 -o $(MODEL_3B) $(MODEL_3B_URL)
	[ -f $(MODEL_15B) ] || curl -fSL --retry 3 -o $(MODEL_15B) $(MODEL_15B_URL)

dataset:
	python3 eval/make_dataset.py

label-3b:
	$(PY) eval/label_local.py --model-path $(MODEL_3B) --tag 3b_2cpu \
	  --threads 2 --budget 18

label-15b:
	$(PY) eval/label_local.py --model-path $(MODEL_15B) --tag $(ROUTER_TAG) \
	  --threads 2 --budget 18

# Final compact router: exact same JSON scorer runs in the image and browser.
train-router:
	python3 router/train_compact_router.py \
	  --labels $(ROUTER_LABELS) \
	  --require-tasks $(ROUTER_TASKS) \
	  --exclude-tasks test_input/tasks.json test_input_19/tasks.json \
	  --out $(ROUTER_CORE) \
	  --demo-out $(ROUTER_DEMO) \
	  --esc-tokens 168
	$(ROUTER_ARTIFACT_CHECK)

# Optional research comparison with the tutorial's 66M-parameter DistilBERT.
train-distilbert:
	$(PY) router/train_router.py --labels data/labels_$(ROUTER_TAG).jsonl \
	  --stats data/category_stats_$(ROUTER_TAG).json --out router_model_distilbert \
	  --device auto --esc-tokens 168

router-artifact-check:
	$(ROUTER_ARTIFACT_CHECK)

router-check: router-artifact-check
	$(PY) router/infer_router.py --tasks test_input_19/tasks.json --model router_model

pick-model:
	$(PY) eval/pick_escalation_model.py

check-models:
	python3 eval/check_models.py

# ---- test & judge ----

test:
	python3 -m pytest tests/ -q

testset:
	python3 eval/make_testset.py

run-local:
	bash eval/run_local.sh $(IMAGE)

# 19-task timing rehearsal under judge-like limits (4g/2cpu, real deadlines).
rehearsal:
	bash eval/run_local.sh $(IMAGE) test_input_19

# 80-task accuracy pass with deadlines lifted.
accuracy:
	bash eval/run_local.sh $(IMAGE) test_input --accuracy

judge:
	python3 eval/judge.py

judge-19:
	python3 eval/judge.py test_input_19

all: test build testset run-local judge
