IMAGE ?= yourdockerhubuser/gp-agent:latest
DEV_IMAGE ?= gp-agent:dev-arm

.PHONY: build build-dev build-remote build-push push test testset run-local judge check-models all

build:
	docker buildx build --platform linux/amd64 -t $(IMAGE) --load .

# Native Apple Silicon dev build — no qemu emulation, real tok/s for iteration.
build-dev:
	docker buildx build --platform linux/arm64 -t $(DEV_IMAGE) --load .

# Lightweight accuracy-first fallback when publishing the bundled GGUF is slow.
build-remote:
	docker buildx build --platform linux/amd64 -f Dockerfile.remote -t $(IMAGE) --load .

# Final submission build+push (always amd64).
build-push:
	docker buildx build --platform linux/amd64 -t $(IMAGE) --push .

push: build-push

check-models:
	python3 eval/check_models.py

test:
	python3 -m pytest tests/ -q

testset:
	python3 eval/make_testset.py

run-local:
	bash eval/run_local.sh $(IMAGE)

judge:
	python3 eval/judge.py

all: test build testset run-local judge
