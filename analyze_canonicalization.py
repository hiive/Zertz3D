#!/usr/bin/env python3
"""
Analyze board state canonicalization from notation files.

Shows original and canonicalized forms of each state, plus statistics
about how many original states map to each canonical state.
"""

import argparse
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np
from PIL import Image
import tempfile

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from game.utils.diagram import execute_notation_sequence, DiagramRenderer
from game.loaders import NotationLoader


def state_to_key(state: np.ndarray) -> str:
    """Convert state array to hashable key for dictionary."""
    return state.tobytes()


def analyze_canonicalization(notation_path: str, save_images: bool = False, output_dir: str = None, show_all_transforms: bool = False, image_columns: int = 6):
    """Analyze canonicalization from a notation file.

    Args:
        notation_path: Path to notation file
        save_images: If True, save visualization images
        output_dir: Directory to save images (default: canonicalization_output)
        show_all_transforms: If True, show all unique transformations in images
        image_columns: Number of images per row in output (default: 6)
    """

    # Load notation to count moves
    loader = NotationLoader(notation_path)
    player1_actions, player2_actions = loader.load()

    # Total moves = sum of both players' actions
    total_moves = len(player1_actions) + len(player2_actions)

    print(f"Analyzing canonicalization for {notation_path}")
    print(f"Board size: {loader.detected_rings}")
    print(f"Variant: {'Blitz' if loader.blitz else 'Standard'}")
    print(f"Total moves: {total_moves}")
    print("=" * 80)
    print()

    # Track states
    canonical_counts = defaultdict(int)  # canonical_key -> count
    canonical_first_seen = {}  # canonical_key -> move_num
    canonical_to_id = {}  # canonical_key -> sequential ID
    move_info = []  # List of (move_num, original_state, canonical_state, transform, original_board)

    # Prepare output directory if saving images
    if save_images:
        if output_dir:
            base_dir = Path(output_dir)
        else:
            base_dir = Path("canonicalization_output")
        base_dir.mkdir(parents=True, exist_ok=True)
        renderer = DiagramRenderer(show_removed=True, edge_coords=True)
        print(f"Saving visualizations to {base_dir}/")
        print()

    # Analyze each state
    for move_num in range(total_moves + 1):  # +1 for initial state
        # Execute notation up to this move
        # stop_at_move is 0-indexed number of moves to execute
        # move_num=0 -> no moves (initial state)
        # move_num=1 -> 1 move executed
        board = execute_notation_sequence(notation_path, stop_at_move=move_num if move_num > 0 else 0)

        # Get states
        original_state = board.state.copy()
        canonical_state, transform, _ = board.canonicalize_state()

        # Track canonical state
        canonical_key = state_to_key(canonical_state)
        canonical_counts[canonical_key] += 1

        if canonical_key not in canonical_first_seen:
            canonical_first_seen[canonical_key] = move_num
            canonical_to_id[canonical_key] = len(canonical_to_id) + 1

        # Store for later display
        move_info.append((move_num, original_state, canonical_state, transform, board))

    print(f"Total states analyzed: {len(move_info)}")
    print(f"Unique canonical states: {len(canonical_counts)}")
    print()

    # Display each state
    print("=" * 80)
    print("ORIGINAL vs CANONICALIZED STATES")
    print("=" * 80)
    print()

    if save_images:
        seed_dir = base_dir / Path(notation_path).stem
        seed_dir.mkdir(exist_ok=True)
        print(f"Saving visualizations to {seed_dir}/")

    for move_num, original_state, canonical_state, transform, board in move_info:
        is_canonical = np.array_equal(original_state, canonical_state)
        canonical_key = state_to_key(canonical_state)
        canonical_id = canonical_to_id[canonical_key]

        print(f"Move {move_num}:")
        print(f"  Status: {'ALREADY CANONICAL' if is_canonical else f'TRANSFORMED (via {transform})'}")
        print(f"  Canonical State ID: #{canonical_id}")

        if save_images:
            from game.zertz_board import ZertzBoard

            # Collect all images to concatenate
            images_to_concat = []
            temp_files = []

            # 1. Canonical state (left-most)
            tmp_canon = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
            temp_files.append(tmp_canon.name)
            canonical_board = ZertzBoard(loader.detected_rings)
            canonical_board.state = canonical_state.copy()
            renderer.save_board(canonical_board, tmp_canon.name, title=f"Canonical (R0)", width=512)
            images_to_concat.append(Image.open(tmp_canon.name))

            # 2. Original state
            tmp_orig = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
            temp_files.append(tmp_orig.name)
            renderer.save_board(board, tmp_orig.name, title=f"Original ({transform})", width=512)
            images_to_concat.append(Image.open(tmp_orig.name))

            # 3. Optionally add all other unique transformations
            if show_all_transforms:
                # Get all unique transformations of the original state
                all_transforms = board.canonicalizer.get_all_transformations(
                    state=original_state,
                    include_translation=False
                )

                # Filter out canonical and original (which is identity if already canonical)
                original_key = state_to_key(original_state)
                canonical_key = state_to_key(canonical_state)

                for transform_name, transformed_state in sorted(all_transforms.items()):
                    transformed_key = state_to_key(transformed_state)

                    # Skip if it's the canonical or original state
                    if transformed_key == canonical_key or transformed_key == original_key:
                        continue

                    # Render this transformation
                    tmp_transform = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
                    temp_files.append(tmp_transform.name)

                    transform_board = ZertzBoard(loader.detected_rings)
                    transform_board.state = transformed_state.copy()
                    renderer.save_board(transform_board, tmp_transform.name, title=transform_name, width=512)
                    images_to_concat.append(Image.open(tmp_transform.name))

            # Arrange images in a grid with specified number of columns
            num_images = len(images_to_concat)
            num_cols = min(image_columns, num_images)  # Don't exceed actual number of images
            num_rows = (num_images + num_cols - 1) // num_cols  # Ceiling division

            # Calculate grid dimensions
            img_width = images_to_concat[0].width if images_to_concat else 0
            img_height = images_to_concat[0].height if images_to_concat else 0
            total_width = num_cols * img_width
            total_height = num_rows * img_height

            # Get background color from first image (top-left pixel)
            bg_color = images_to_concat[0].getpixel((0, 0)) if images_to_concat else (255, 255, 255)
            combined_img = Image.new('RGB', (total_width, total_height), bg_color)

            # Paste images in grid layout
            for idx, img in enumerate(images_to_concat):
                row = idx // num_cols
                col = idx % num_cols
                x_offset = col * img_width
                y_offset = row * img_height
                combined_img.paste(img, (x_offset, y_offset))

            # Save combined image
            combined_filename = f"move_{move_num:03d}_{transform}.png"
            combined_path = seed_dir / combined_filename
            combined_img.save(combined_path)

            # Clean up temp files
            for temp_file in temp_files:
                Path(temp_file).unlink()

        print("-" * 80)

    # Display summary table
    print()
    print("=" * 80)
    print("CANONICALIZATION SUMMARY")
    print("=" * 80)
    print()
    print(f"{'Canonical State ID':<25} {'First Seen':<15} {'Source States':<20} {'Permutations':<20}")
    print("-" * 80)

    # Sort by first seen
    sorted_canonical = sorted(canonical_first_seen.items(), key=lambda x: x[1])

    # Track permutations for average calculation
    all_permutation_counts = []

    for canonical_key, first_seen in sorted_canonical:
        canonical_id = canonical_to_id[canonical_key]
        count = canonical_counts[canonical_key]

        # Get the board for this canonical state to count permutations
        # Find a move that has this canonical state
        num_permutations = 0
        for move_num, original_state, canonical_state, transform, board in move_info:
            if state_to_key(canonical_state) == canonical_key:
                # Count all valid transformations of this canonical state
                all_transforms = board.canonicalizer.get_all_transformations(
                    state=canonical_state,
                    include_translation=False  # Only count rotation/mirror symmetries
                )
                num_permutations = len(all_transforms)
                all_permutation_counts.append(num_permutations)
                break

        print(f"{f'State #{canonical_id}':<25} {f'Move {first_seen}':<15} {count:<20} {num_permutations:<20}")

    # Calculate average permutations
    avg_permutations = sum(all_permutation_counts) / len(all_permutation_counts) if all_permutation_counts else 0

    print("-" * 80)
    print(f"{'TOTAL':<25} {'':<15} {len(move_info):<20} {f'{avg_permutations:.1f} avg':<20}")
    print()

    # Compression ratio - useful for multi-game analysis, but not for single game sequences
    # where each move typically creates a unique state (ratio will almost always be 1.0)
    # Uncomment when analyzing multiple games or branching game trees:
    # compression_ratio = len(move_info) / len(canonical_counts) if canonical_counts else 1.0
    # print(f"Compression ratio: {compression_ratio:.2f}x")
    # print(f"  ({len(move_info)} original states -> {len(canonical_counts)} canonical states)")
    # print()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze board state canonicalization from notation files"
    )
    parser.add_argument(
        "notation_file",
        help="Path to notation file"
    )
    parser.add_argument(
        "--save-images",
        action="store_true",
        help="Save visualization images for each state"
    )
    parser.add_argument(
        "--show-all-transforms",
        action="store_true",
        help="Show all unique transformations in saved images (implies --save-images)"
    )
    parser.add_argument(
        "--output-dir",
        help="Directory to save images (default: canonicalization_output)"
    )
    parser.add_argument(
        "--image-columns",
        type=int,
        default=6,
        help="Number of images per row in grid layout (default: 6)"
    )

    args = parser.parse_args()

    if not Path(args.notation_file).exists():
        print(f"Error: File not found: {args.notation_file}")
        sys.exit(1)

    # If --show-all-transforms is set, enable --save-images
    save_images = args.save_images or args.show_all_transforms

    analyze_canonicalization(
        args.notation_file,
        save_images=save_images,
        output_dir=args.output_dir,
        show_all_transforms=args.show_all_transforms,
        image_columns=args.image_columns
    )


if __name__ == "__main__":
    main()