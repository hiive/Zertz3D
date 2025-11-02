#!/usr/bin/env python3
"""
Render an entire game turn-by-turn as a grid of board positions.

Each board is labeled with turn number, player name, and move notation.
"""

import argparse
import sys
from pathlib import Path
import tempfile

# Add project root to Python path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir.parent))
from utils.project_path import find_project_root

project_root = find_project_root(Path(__file__).parent)
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from game.utils.diagram import execute_notation_sequence, DiagramRenderer, action_dict_to_str
from game.utils.grid_renderer import GridRenderer
from game.loaders import AutoSelectLoader
from game.formatters import NotationFormatter
from game.zertz_game import ZertzGame
from game.constants import STANDARD_MARBLES, BLITZ_MARBLES, STANDARD_WIN_CONDITIONS, BLITZ_WIN_CONDITIONS


def render_game_sequence(
    replay_path: str,
    save_images: bool = False,
    output_dir: str = None,
    image_columns: int = 6,
    separate_turns: bool = False,
    svg: bool = False,
    show_captures: bool = False,
    show_row_dividers: bool = False,
):
    """Render a complete game as a grid of turn-by-turn board positions.

    Args:
        replay_path: Path to replay file (notation, transcript, or SGF)
        save_images: If True, save to file; if False, display on screen
        output_dir: Directory to save images (default: game_sequences)
        image_columns: Number of images per row (default: 6, combined mode only)
        separate_turns: If True, save individual files per turn instead of combined grid
        svg: If True, use SVG format (supports both combined grid and separate files)
        show_captures: If True, display player capture counts at bottom corners
        show_row_dividers: If True, draw thin dividers between rows in combined grid (default: False)
    """

    # Load replay file (auto-detects format)
    loader = AutoSelectLoader(replay_path)
    player1_actions, player2_actions = loader.load()

    # Get player names (or use defaults)
    player1_name = loader.player1_name or "Player 1"
    player2_name = loader.player2_name or "Player 2"

    # Total moves = sum of both players' actions
    total_moves = len(player1_actions) + len(player2_actions)

    print(f"Rendering game sequence for {replay_path}")
    print(f"Board size: {loader.detected_rings}")
    print(f"Variant: {'Blitz' if loader.blitz else 'Standard'}")
    print(f"Player 1: {player1_name}")
    print(f"Player 2: {player2_name}")
    print(f"Total moves: {total_moves}")
    print("=" * 80)
    print()

    # Create renderer
    renderer = DiagramRenderer(show_removed=True, edge_coords=True, show_captures=show_captures)

    # Prepare output directory and filenames
    if output_dir:
        base_dir = Path(output_dir)
    else:
        base_dir = Path("game_sequences")

    replay_stem = Path(replay_path).stem

    # Create directory if saving images or separate turns
    if save_images or separate_turns:
        base_dir.mkdir(parents=True, exist_ok=True)

        if separate_turns:
            if svg:
                # For SVG, create subdirectory for individual files
                output_subdir = base_dir / f"{replay_stem}_svg"
                output_subdir.mkdir(exist_ok=True)
                print(f"Saving individual SVG files to: {output_subdir}/")
            else:
                # For separate PNG files
                output_subdir = base_dir / f"{replay_stem}_separate"
                output_subdir.mkdir(exist_ok=True)
                print(f"Saving individual PNG files to: {output_subdir}/")
        else:
            format_name = "SVG" if svg else "PNG"
            print(f"Saving combined {format_name} grid to: {base_dir}/")
        print()

    # Resolution
    board_width = 768

    # Render each position with transparent backgrounds for grid mode
    temp_files = []

    # Render initial position
    print("Rendering initial position...")
    board = execute_notation_sequence(replay_path, stop_at_move=0)

    if separate_turns:
        # Save directly to file (SVG or PNG) with solid background
        file_ext = ".svg" if svg else ".png"
        output_path = output_subdir / f"move_000_initial{file_ext}"
        renderer.save_board(board, output_path, title="Initial Position", width=board_width, transparent=False)
        print(f"  Saved: {output_path.name}")
    else:
        # Save to temp file for grid with transparent background
        file_ext = '.svg' if svg else '.png'
        tmp_file = tempfile.NamedTemporaryFile(suffix=file_ext, delete=False)
        temp_files.append(tmp_file.name)
        renderer.save_board(board, tmp_file.name, title="Initial Position", width=board_width, transparent=True)

    # Build list of all moves in sequence with metadata
    marbles = BLITZ_MARBLES if loader.blitz else STANDARD_MARBLES
    win_condition = BLITZ_WIN_CONDITIONS if loader.blitz else STANDARD_WIN_CONDITIONS
    game = ZertzGame(loader.detected_rings, marbles, win_condition, t=5)

    # Build move sequence with correct turn tracking
    move_sequence = []  # List of (move_num, player_name, action_dict, notation)
    p1_idx = 0
    p2_idx = 0

    for move_num in range(1, total_moves + 1):
        # Check whose turn it is
        current_player = game.get_cur_player_value()

        # Get the action for the current player
        if current_player == 1:
            if p1_idx >= len(player1_actions):
                print(f"Warning: Player 1 out of actions at move {move_num}")
                break
            action_dict = player1_actions[p1_idx]
            player_name = player1_name
            p1_idx += 1
        else:
            if p2_idx >= len(player2_actions):
                print(f"Warning: Player 2 out of actions at move {move_num}")
                break
            action_dict = player2_actions[p2_idx]
            player_name = player2_name
            p2_idx += 1

        # Apply the action to get ActionResult
        action_str = action_dict_to_str(action_dict)
        action_type, action = game.str_to_action(action_str)
        action_result = game.take_action(action_type, action)

        # Convert to notation
        notation = NotationFormatter.action_to_notation(action_dict, action_result)

        # Store move info
        move_sequence.append((move_num, player_name, action_dict, notation))

    # Now render each position with correct captions
    for move_num, player_name, action_dict, notation in move_sequence:
        # Render board after this move
        print(f"Rendering move {move_num}: {player_name} - {notation}")
        board = execute_notation_sequence(replay_path, stop_at_move=move_num)

        # Create title with turn, player, and notation
        title = f"Turn {move_num}: {player_name}\n{notation}"

        if separate_turns:
            # Save directly to file (SVG or PNG) with solid background
            notation_clean = notation.replace(" ", "_").replace(",", "").replace("x", "")
            file_ext = ".svg" if svg else ".png"
            output_path = output_subdir / f"move_{move_num:03d}_{notation_clean}{file_ext}"
            renderer.save_board(board, output_path, title=title, width=board_width, transparent=False)
            print(f"  Saved: {output_path.name}")
        else:
            # Save to temp file for grid with transparent background
            file_ext = '.svg' if svg else '.png'
            tmp_file = tempfile.NamedTemporaryFile(suffix=file_ext, delete=False)
            temp_files.append(tmp_file.name)
            renderer.save_board(board, tmp_file.name, title=title, width=board_width, transparent=True)

    # Create combined grid using GridRenderer
    if not separate_turns:
        print()
        print(f"Creating grid: {len(temp_files)} total images")

        # Create grid renderer with background color from DiagramRenderer
        grid_renderer = GridRenderer(image_columns=image_columns, background_color=renderer.bg_color, show_row_dividers=show_row_dividers)

        # Create combined grid and clean up temp files
        if save_images:
            output_path = base_dir / f"{replay_stem}_sequence.{('svg' if svg else 'png')}"
            grid_renderer.create_grid_from_temp_files(temp_files, str(output_path), svg=svg)
            print(f"\nSaved game sequence to: {output_path}")
        else:
            if svg:
                print("\nWarning: SVG display not supported. Use --save-images to save to file.")
                # Still need to clean up temp files
                for temp_file in temp_files:
                    try:
                        Path(temp_file).unlink()
                    except Exception:
                        pass
            else:
                # For PNG display: create temp grid file, display it, then clean up
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_grid:
                    grid_output_path = tmp_grid.name

                grid_renderer.create_grid_from_temp_files(temp_files, grid_output_path, svg=False)
                print("\nDisplaying game sequence...")
                from PIL import Image
                Image.open(grid_output_path).show()
                # Clean up the grid file after displaying
                Path(grid_output_path).unlink()

    print("Done!")


def main():
    parser = argparse.ArgumentParser(
        description="Render a complete game as a turn-by-turn sequence"
    )
    parser.add_argument(
        "replay_file",
        help="Path to replay file (notation, transcript, or SGF format)"
    )
    parser.add_argument(
        "--save-images",
        action="store_true",
        help="Save image to file (default: display on screen)"
    )
    parser.add_argument(
        "--output-dir",
        help="Directory to save images (default: game_sequences)"
    )
    parser.add_argument(
        "--image-columns",
        type=int,
        default=6,
        help="Number of board images per row in grid (default: 6, combined mode only)"
    )
    parser.add_argument(
        "--separate-turns",
        action="store_true",
        help="Save individual files per turn instead of combined grid"
    )
    parser.add_argument(
        "--svg",
        action="store_true",
        help="Use SVG format instead of PNG (supports both combined grid and separate files)"
    )
    parser.add_argument(
        "--show-captures",
        action="store_true",
        help="Display player capture counts at bottom corners"
    )
    parser.add_argument(
        "--show-row-dividers",
        action="store_true",
        help="Draw thin dividers between rows in combined grid"
    )

    args = parser.parse_args()

    render_game_sequence(
        args.replay_file,
        save_images=args.save_images,
        output_dir=args.output_dir,
        image_columns=args.image_columns,
        separate_turns=args.separate_turns,
        svg=args.svg,
        show_captures=args.show_captures,
        show_row_dividers=args.show_row_dividers,
    )


if __name__ == "__main__":
    main()