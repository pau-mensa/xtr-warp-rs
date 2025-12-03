.PHONY: install-gpu install clean

install-gpu:
	@echo "Installing..."
	cargo clean
	uv venv
	uv pip install numpy
	uv pip install torch==2.8.0 #--index-url https://download.pytorch.org/whl/cu130
	LIBTORCH=$$(uv run python -c "import torch; import os; print(os.path.dirname(torch.__file__))") \
	LIBTORCH_USE_PYTORCH=1 \
	LIBTORCH_BYPASS_VERSION_CHECK=1 \
	LD_LIBRARY_PATH="$${LIBTORCH}/lib:$${LD_LIBRARY_PATH}" \
	uv pip install -e ".[dev]"

install:
	@echo "Installing..."
	cargo clean
	uv venv
	uv pip install numpy
	uv pip install torch --index-url https://download.pytorch.org/whl/cpu
	LIBTORCH=$$(uv run python -c "import torch; import os; print(os.path.dirname(torch.__file__))") \
	LIBTORCH_USE_PYTORCH=1 \
	LIBTORCH_BYPASS_VERSION_CHECK=1 \
	LD_LIBRARY_PATH="$${LIBTORCH}/lib:$${LD_LIBRARY_PATH}" \
	uv pip install -e ".[dev]"

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
	LIBTORCH=$$(uv run python -c "import torch; import os; print(os.path.dirname(torch.__file__))") \
	LIBTORCH_USE_PYTORCH=1 \
	LIBTORCH_BYPASS_VERSION_CHECK=1 \
	LD_LIBRARY_PATH="$${LIBTORCH}/lib:$${LD_LIBRARY_PATH}" \
	maturin develop --release
