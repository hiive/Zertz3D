"""Backend detection and management for MCTS.

Provides automatic detection of available backends (Python or Rust)
and utilities for switching between them.
"""

import os
from enum import Enum
from typing import Optional

# Try to import Rust backend
HAS_RUST = False
try:
    import hiivelabs_zertz_mcts
    HAS_RUST = True
except ImportError:
    pass


class Backend(Enum):
    """Available MCTS backends."""
    PYTHON = "python"
    RUST = "rust"


def get_backend(requested: Optional[str] = None) -> Backend:
    """Determine which backend to use.

    Args:
        requested: 'python', 'rust', 'auto', or None (checks ZERTZ_BACKEND env var)

    Returns:
        Backend enum value

    Raises:
        ValueError: If Rust backend requested but not available
    """
    backend_str = requested or os.environ.get('ZERTZ_BACKEND', 'auto')
    backend_str = backend_str.lower()

    if backend_str == 'rust':
        if not HAS_RUST:
            raise ValueError(
                "Rust backend requested but not available. "
                "Build with: cd rust && maturin develop --release"
            )
        return Backend.RUST
    elif backend_str == 'python':
        return Backend.PYTHON
    elif backend_str == 'auto':
        return Backend.RUST if HAS_RUST else Backend.PYTHON
    else:
        raise ValueError(
            f"Invalid backend: {backend_str}. "
            f"Must be 'python', 'rust', or 'auto'"
        )


def backend_available(backend: Backend) -> bool:
    """Check if a specific backend is available.

    Args:
        backend: Backend to check

    Returns:
        True if backend is available
    """
    if backend == Backend.PYTHON:
        return True
    elif backend == Backend.RUST:
        return HAS_RUST
    return False


def get_backend_info() -> dict:
    """Get information about available backends.

    Returns:
        Dictionary with backend availability and current selection
    """
    current = get_backend()
    return {
        'python_available': True,
        'rust_available': HAS_RUST,
        'current': current.value,
        'default': 'rust' if HAS_RUST else 'python',
    }