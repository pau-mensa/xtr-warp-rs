.PHONY: help install-gpu install clean build test

help:	## Show all Makefile targets.
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[33m%-30s\033[0m %s\n", $$1, $$2}'

install-gpu:	## Install dependencies for gpu
	@echo "Installing GPU dependencies..."
	@test -d .venv || uv venv
	uv pip install torch --index-url https://download.pytorch.org/whl/cu130
	LIBTORCH_USE_PYTORCH=1 LIBTORCH_BYPASS_VERSION_CHECK=1 uv pip install --no-build-isolation -e .[dev]

install:	## Install dependencies for cpu
	@echo "Installing CPU dependencies..."
	@test -d .venv || uv venv
	uv pip install torch --index-url https://download.pytorch.org/whl/cpu
	LIBTORCH_USE_PYTORCH=1 LIBTORCH_BYPASS_VERSION_CHECK=1 uv pip install --no-build-isolation -e .[dev]

clean:	## Clean build artifacts
	cargo clean
	rm -rf target/
	rm -rf build/
	rm -rf *.egg-info
	rm -rf dist/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

build:	## Build the project
	export LIBTORCH=$$(uv run python -c "import torch; import os; print(os.path.dirname(torch.__file__))") && \
	export LIBTORCH_USE_PYTORCH=1 && \
	export LIBTORCH_BYPASS_VERSION_CHECK=1 && \
	export CXXFLAGS="-w" && \
	uv run maturin develop --release

test:	## Run tests
	uv run pytest tests/test.py
