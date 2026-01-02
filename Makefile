.PHONY: install-gpu install clean build test

# Note: Due to PyTorch + Rust binding requirements, tests must be run with 'make test'
# rather than 'uv run pytest' to ensure proper environment variables are set.
# The chain is: make clean && make install-gpu && make build && make test

install-gpu:
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

install:
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

clean:
	cargo clean
	rm -rf target/
	rm -rf build/
	rm -rf *.egg-info
	rm -rf dist/
	rm -rf .venv
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

build:
	@. .venv/bin/activate && \
	export LIBTORCH=$$(python -c "import torch; import os; print(os.path.dirname(torch.__file__))") && \
	export LIBTORCH_USE_PYTORCH=1 && \
	export LIBTORCH_BYPASS_VERSION_CHECK=1 && \
	export LD_LIBRARY_PATH="$${LIBTORCH}/lib:$${LD_LIBRARY_PATH}" && \
	export CXXFLAGS="-w" && \
	maturin develop --release

test:
	@. .venv/bin/activate && \
	export LIBTORCH=$$(python -c "import torch; import os; print(os.path.dirname(torch.__file__))") && \
	export LIBTORCH_USE_PYTORCH=1 && \
	export LIBTORCH_BYPASS_VERSION_CHECK=1 && \
	export LD_LIBRARY_PATH="$${LIBTORCH}/lib:$${LD_LIBRARY_PATH}" && \
	pytest tests/test.py
