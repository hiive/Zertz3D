#!/usr/bin/env python3
"""Command-line utility to render Zèrtz board states from notation.

Examples:
    # Render notation file to PNG
    python render_notation.py game.txt --output board.png

    # Display on screen with coordinates
    python render_notation.py game.txt --show --coords

    # Render specific move number
    python render_notation.py game.txt --output move_15.png --stop-at 15

    # Custom size and title
    python render_notation.py game.txt --output final.png --width 1200 --title "Final Position"
"""

import sys
from pathlib import Path

# Add project root to Python path to support running from any directory
def find_project_root(start_path: Path) -> Path:
    """Find project root by searching for pyproject.toml."""
    current = start_path.resolve()
    while current != current.parent:
        if (current / 'pyproject.toml').exists():
            return current
        current = current.parent
    raise RuntimeError("Could not find project root (pyproject.toml not found)")

project_root = find_project_root(Path(__file__).parent)
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import argparse
from pathlib import Path
from game.utils.diagram import render_board_from_notation


def main():
    parser = argparse.ArgumentParser(
        description="Render Zèrtz board states from notation files or strings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "notation",
        help="Path to notation file or notation string"
    )

    parser.add_argument(
        "-o", "--output",
        help="Output file path (PNG recommended). If not specified, displays on screen."
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the image on screen (in addition to saving if --output is specified)"
    )

    parser.add_argument(
        "--coords",
        action="store_true",
        help="Show coordinate labels on all rings"
    )

    parser.add_argument(
        "--edge-coords",
        action="store_true",
        help="Show coordinate labels on top/bottom edge rings only"
    )

    parser.add_argument(
        "--show-removed",
        action="store_true",
        help="Show removed rings with transparency"
    )

    parser.add_argument(
        "--title",
        help="Title for the figure"
    )

    parser.add_argument(
        "--width",
        type=int,
        help="Output width in pixels (default: 1500, height auto-calculated if omitted)"
    )

    parser.add_argument(
        "--height",
        type=int,
        help="Output height in pixels (default: 1500, width auto-calculated if omitted)"
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Resolution in dots per inch for saved image (default: 150)"
    )

    parser.add_argument(
        "--stop-at",
        type=int,
        help="Stop at this move number (0-indexed)"
    )

    parser.add_argument(
        "--bg-color",
        help="Background color in #RRGGBB or #RRGGBBAA format (default: #F5E6D3)"
    )

    args = parser.parse_args()

    # Convert notation to Path if it's a file
    notation_input = args.notation
    path = Path(notation_input)
    if path.exists() and path.is_file():
        print(f"Loading notation from: {path}")
    else:
        print("Interpreting input as notation string")

    # Render board
    try:
        render_board_from_notation(
            notation_input,
            output_path=args.output,
            show=args.show,
            show_coords=args.coords,
            edge_coords=args.edge_coords,
            show_removed=args.show_removed,
            title=args.title,
            width=args.width,
            height=args.height,
            dpi=args.dpi,
            stop_at_move=args.stop_at,
            bg_color=args.bg_color,
        )

        if args.output:
            print(f"Saved board image to: {args.output}")

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())