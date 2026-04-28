VENV    := .venv
PYTHON  := $(VENV)/bin/python
MATURIN := $(VENV)/bin/maturin
PYTEST  := $(VENV)/bin/pytest

# Common env block for recipes that need torch / libtorch.
# Exported inline so cargo subprocesses (torch-sys build script) inherit them.
TORCH_ENV = \
	VIRTUAL_ENV=$(CURDIR)/$(VENV) \
	PATH=$(CURDIR)/$(VENV)/bin:$$PATH \
	LIBTORCH_USE_PYTORCH=1 \
	LIBTORCH_BYPASS_VERSION_CHECK=1 \
	LIBTORCH=$$($(PYTHON) -c "import torch,os;print(os.path.dirname(torch.__file__))")

.PHONY: help install-gpu install clean build test test-all

help:	## Show all Makefile targets.
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[33m%-30s\033[0m %s\n", $$1, $$2}'

install-gpu:	## Install dependencies for gpu
	@echo "Installing GPU dependencies..."
	@test -d $(VENV) || uv venv $(VENV)
	$(TORCH_ENV) uv pip install torch --index-url https://download.pytorch.org/whl/cu130
	$(TORCH_ENV) uv pip install --no-build-isolation -e .[dev]

install:	## Install dependencies for cpu
	@echo "Installing CPU dependencies..."
	@test -d $(VENV) || uv venv $(VENV)
	$(TORCH_ENV) uv pip install torch --index-url https://download.pytorch.org/whl/cpu
	$(TORCH_ENV) uv pip install --no-build-isolation -e .[dev]

clean:	## Clean build artifacts
	cargo clean
	rm -rf target/
	rm -rf build/
	rm -rf *.egg-info
	rm -rf dist/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

build:	## Build the project
	@test -x $(PYTHON) || { echo "No venv found — run 'make install' first"; exit 1; }
	$(TORCH_ENV) CXXFLAGS="-w" $(MATURIN) develop --release

test:	## Run tests (excludes sharding tests)
	@test -x $(PYTEST) || { echo "No venv found — run 'make install' first"; exit 1; }
	$(TORCH_ENV) $(PYTEST) tests/test.py tests/test_index_management.py tests/test_filtering.py

test-all:	## Run all tests including sharding
	@test -x $(PYTEST) || { echo "No venv found — run 'make install' first"; exit 1; }
	$(TORCH_ENV) $(PYTEST) tests/ -m ""
