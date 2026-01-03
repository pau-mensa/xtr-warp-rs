.PHONY: help install-gpu install clean build test

help:	## Show all Makefile targets.
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[33m%-30s\033[0m %s\n", $$1, $$2}'

install-gpu:	## Install dependencies for gpu
	@echo "Installing..."
	cargo clean
	uv venv
	uv pip install numpy
	uv pip install torch --index-url https://download.pytorch.org/whl/cu130
	uv pip install maturin pytest-cov pytest ruff pre-commit beir ranx fastkmeans joblib setuptools
	@echo "Building with maturin..."
	@. .venv/bin/activate && \
	export LIBTORCH=$$(python -c "import torch; import os; print(os.path.dirname(torch.__file__))") && \
	export LIBTORCH_USE_PYTORCH=1 && \
	export LIBTORCH_BYPASS_VERSION_CHECK=1 && \
	export LD_LIBRARY_PATH="$${LIBTORCH}/lib:$${LD_LIBRARY_PATH}" && \
	export CXXFLAGS="-w" && \
	maturin develop --release

install:	## Install dependencies for cpu
	@echo "Installing..."
	cargo clean
	uv venv
	uv pip install numpy
	uv pip install torch --index-url https://download.pytorch.org/whl/cpu
	uv pip install maturin pytest-cov pytest ruff pre-commit beir ranx fastkmeans joblib setuptools
	@echo "Building with maturin..."
	@. .venv/bin/activate && \
	export LIBTORCH=$$(python -c "import torch; import os; print(os.path.dirname(torch.__file__))") && \
	export LIBTORCH_USE_PYTORCH=1 && \
	export LIBTORCH_BYPASS_VERSION_CHECK=1 && \
	export LD_LIBRARY_PATH="$${LIBTORCH}/lib:$${LD_LIBRARY_PATH}" && \
	export CXXFLAGS="-w" && \
	maturin develop --release

clean:	## Clean build artifacts
	cargo clean
	rm -rf target/
	rm -rf build/
	rm -rf *.egg-info
	rm -rf dist/
	rm -rf .venv
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

build:	## Build the project
	@. .venv/bin/activate && \
	export LIBTORCH=$$(python -c "import torch; import os; print(os.path.dirname(torch.__file__))") && \
	export LIBTORCH_USE_PYTORCH=1 && \
	export LIBTORCH_BYPASS_VERSION_CHECK=1 && \
	export LD_LIBRARY_PATH="$${LIBTORCH}/lib:$${LD_LIBRARY_PATH}" && \
	export CXXFLAGS="-w" && \
	maturin develop --release

test:	## Run tests
	@. .venv/bin/activate && \
	export LIBTORCH=$$(python -c "import torch; import os; print(os.path.dirname(torch.__file__))") && \
	export LIBTORCH_USE_PYTORCH=1 && \
	export LIBTORCH_BYPASS_VERSION_CHECK=1 && \
	export LD_LIBRARY_PATH="$${LIBTORCH}/lib:$${LD_LIBRARY_PATH}" && \
	pytest tests/test.py
