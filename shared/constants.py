"""Shared constants for cross-layer coordination."""

from __future__ import annotations

# Marble ordering used in masks and supply pools (index 0=white, 1=gray, 2=black)
MARBLE_TYPES: tuple[str, str, str] = ("w", "g", "b")

# Renderer highlight context names for supply marbles
SUPPLY_CONTEXT_MAP: dict[str, str] = {
    "w": "supply_w",
    "g": "supply_g",
    "b": "supply_b",
}
