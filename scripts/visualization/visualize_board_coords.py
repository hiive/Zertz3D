#!/usr/bin/env python3
"""
Visualize a Zèrtz board, highlighting the rings nearest to the geometric center.

Supports all board sizes:
- 37 rings (D6 symmetry, regular hexagon)
- 48 rings (D3 symmetry, side lengths 5-4-5-4-5-4)
- 61 rings (D6 symmetry, regular hexagon)

- Computes the geometric center as the mean of all ring coordinates.
- Highlights any rings whose centers are within a small distance tolerance
  of the nearest-to-center distance.
- Rotates the layout 30° counter-clockwise so a point faces upward.
"""

import sys
from pathlib import Path

# Add project root to Python path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir.parent))
from utils.project_path import find_project_root

project_root = find_project_root(Path(__file__).parent)
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import sys

sys.path.insert(0, "game")

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from game.zertz_board import ZertzBoard


# --------------------------------------------------------------------------- #
# Coordinate conversion helpers
# --------------------------------------------------------------------------- #


def yx_to_axial(y, x, width):
    """Zertz axial convention: q = x - center, r = y - x."""
    c = width // 2
    q = x - c
    r = y - x
    return q, r


def axial_to_cart(q, r, size=1.0):
    """Convert axial (q,r) to Cartesian for a pointy-top hex grid."""
    xc = size * np.sqrt(3) * (q + r / 2.0)
    yc = size * 1.5 * r
    return np.array([xc, yc])


# --------------------------------------------------------------------------- #
# Visualization
# --------------------------------------------------------------------------- #


def visualize_board(rings=37):
    """
    Visualize a Zèrtz board with the specified number of rings.

    Args:
        rings: Number of rings on the board (37, 48, or 61)
    """
    board = ZertzBoard(rings)
    W = board.width
    size = 1.0
    ring_radius = 0.866 * size

    # Determine symmetry type for title
    if rings == ZertzBoard.MEDIUM_BOARD_48:
        symmetry = "D3 symmetry"
    else:
        symmetry = "D6 symmetry"

    fig, ax = plt.subplots(figsize=(10, 9))
    ax.set_aspect("equal")
    ax.grid(False)

    # Collect all ring coordinates
    positions = []
    for y in range(W):
        for x in range(W):
            if board.state[board.RING_LAYER, y, x] == 1:
                label = board.index_to_str((y, x))
                q, r = yx_to_axial(y, x, W)
                cart = axial_to_cart(q, r, size)
                positions.append((label, (y, x), (q, r), cart))

    # Compute geometric center
    all_coords = np.stack([p[3] for p in positions])
    geometric_center = np.mean(all_coords, axis=0)

    # --------------------------------------------------------------- #
    # Apply 30° CCW rotation about the geometric center
    # --------------------------------------------------------------- #
    theta = np.deg2rad(30)
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    rotated_coords = (all_coords - geometric_center) @ R.T + geometric_center

    # --------------------------------------------------------------- #
    # Flip Y so numbering increases upward (A1 bottom)
    # --------------------------------------------------------------- #
    rotated_coords[:, 1] *= -1
    geometric_center[1] *= -1

    # Replace positions' Cartesian coords with rotated versions
    positions = [
        (label, yx, qr, rc) for (label, yx, qr, _), rc in zip(positions, rotated_coords)
    ]
    all_coords = rotated_coords
    # --------------------------------------------------------------- #

    # Compute distances to center
    dists = np.linalg.norm(all_coords - geometric_center, axis=1)
    min_dist = np.min(dists)
    tolerance = 1e-6  # nearly exact match in this layout
    near_center_idxs = np.where(np.isclose(dists, min_dist, atol=tolerance))[0]

    # Determine which rings are closest to center
    closest_rings = [positions[i] for i in near_center_idxs]
    closest_labels = [r[0] for r in closest_rings]

    # Draw all rings
    for label, (_, _), (_, _), (xc, yc) in positions:
        if label in closest_labels:
            color = "red"
            lw = 2.8
        else:
            color = "blue"
            lw = 1.3
        ax.add_patch(
            Circle((xc, yc), ring_radius, fill=False, edgecolor=color, linewidth=lw)
        )
        ax.text(xc, yc, label, ha="center", va="center", fontsize=8, fontweight="bold")

    # Draw geometric center as red star
    ax.plot(
        geometric_center[0],
        geometric_center[1],
        "r*",
        markersize=16,
        label="Geometric Center",
    )

    # Formatting
    pad = 2.2 * size
    ax.set_xlim(all_coords[:, 0].min() - pad, all_coords[:, 0].max() + pad)
    ax.set_ylim(all_coords[:, 1].min() - pad, all_coords[:, 1].max() + pad)
    ax.set_xlabel("X (axial→Cartesian, pointy-top)")
    ax.set_ylabel("Y (axial→Cartesian, pointy-top)")
    ax.set_title(
        f"{rings}-Ring Zèrtz Board ({symmetry}) — Rings Closest to Geometric Center",
        fontsize=13,
        fontweight="bold",
    )

    # Add legend with names of center-nearest rings
    ring_text = ", ".join(closest_labels)
    ax.legend(
        [plt.Line2D([], [], color="red", linewidth=2)],
        [f"Closest ring(s): {ring_text}"],
        loc="upper right",
    )

    plt.tight_layout()
    output_path = f"board_visualization_{rings}.png"
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    print(f"\nSaved: {output_path}")
    print(f"Board: {rings} rings ({symmetry})")
    print(f"Geometric Center: ({geometric_center[0]:.4f}, {geometric_center[1]:.4f})")
    print(f"Closest ring(s) to center: {ring_text}")

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize Zèrtz board with geometric center highlighted"
    )
    parser.add_argument(
        "--rings",
        type=int,
        default=37,
        choices=[37, 48, 61],
        help="Number of rings on the board (default: 37)",
    )
    args = parser.parse_args()

    visualize_board(args.rings)
