"""XTR-WARP: A high-performance document retrieval toolkit.

This package provides Python bindings for the Rust implementation of XTR-WARP,
a ColBERT-style late interaction retrieval system.
"""

from .evaluation import evaluate, load_beir
from .search import XTRWarp

try:
    from importlib.metadata import PackageNotFoundError, version as _pkg_version
except ImportError:  # pragma: no cover (py<3.8)
    PackageNotFoundError = Exception  # type: ignore[misc,assignment]
    _pkg_version = None  # type: ignore[assignment]

# Import the Rust extension module
try:
    from . import xtr_warp_rs
except ImportError:
    try:
        import xtr_warp_rs
    except ImportError:
        import warnings

        warnings.warn(
            "xtr_warp_rs module not found. Please build the Rust extension with: "
            "maturin develop --release",
            ImportWarning,
        )
        xtr_warp_rs = None

try:
    __version__ = _pkg_version("xtr-warp-rs") if _pkg_version else "0.0.0"
except PackageNotFoundError:
    __version__ = "0.0.0"
__all__ = ["XTRWarp", "xtr_warp_rs", "evaluate", "load_beir"]
