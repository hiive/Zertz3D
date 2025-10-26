"""Rust backend detection for MCTS.

Ensures the Rust backend is available for MCTS operations.
"""

# Try to import Rust backend
HAS_RUST = False
try:
    import hiivelabs_zertz_mcts
    HAS_RUST = True
except ImportError:
    pass


def ensure_rust_backend():
    """Ensure Rust backend is available.

    Raises:
        RuntimeError: If Rust backend is not available
    """
    if not HAS_RUST:
        raise RuntimeError(
            "Rust backend is required but not available. "
            "Build with: cd rust && maturin develop --release"
        )