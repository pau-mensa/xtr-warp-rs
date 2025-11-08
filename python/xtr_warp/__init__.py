"""XTR-WARP: A high-performance document retrieval toolkit.

This package provides Python bindings for the Rust implementation of XTR-WARP,
a ColBERT-style late interaction retrieval system.
"""

from .search import XTRWarp

# Import the Rust extension module
try:
    from . import xtr_warp_rust
except ImportError:
    try:
        import xtr_warp_rust
    except ImportError:
        import warnings

        warnings.warn(
            "xtr_warp_rust module not found. Please build the Rust extension with: "
            "maturin develop --release",
            ImportWarning,
        )
        xtr_warp_rust = None

__version__ = "0.0.1"
__all__ = ["XTRWarp", "xtr_warp_rust"]
