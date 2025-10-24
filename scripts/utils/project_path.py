"""Utility for finding the project root directory."""

from __future__ import annotations

from pathlib import Path


def find_project_root(start_path: Path) -> Path:
    """
    Find project root by searching for pyproject.toml.

    Args:
        start_path: Starting path to search from (typically __file__ parent)

    Returns:
        Path to project root directory

    Raises:
        RuntimeError: If pyproject.toml is not found in any parent directory
    """
    current = start_path.resolve()
    while current != current.parent:
        if (current / 'pyproject.toml').exists():
            return current
        current = current.parent
    raise RuntimeError("Could not find project root (pyproject.toml not found)")