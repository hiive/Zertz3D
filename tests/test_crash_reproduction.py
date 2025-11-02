"""
Test to reproduce the crash from rust_crash.txt

The crash occurred with:
- Seed: 1760910995
- Player 2 using MCTS with 2500 iterations
- Error: ValueError: Position (0, 6) is not a valid ring coordinate
"""

import pytest
import numpy as np
from game.zertz_game import ZertzGame


def test_crash_seed_1760910995_with_high_iterations():
    """Reproduce crash from log with exact seed and high iterations."""
    import hiivelabs_mcts

    # Exact seed from crash
    np.random.seed(1760910995)
    game = ZertzGame(rings=37)

    # Player 1 move from crash log: PUT w at E1, remove A4
    width = game.board.config.width
    e1_yx = game.board.str_to_index("E1")
    a4_yx = game.board.str_to_index("A4")
    e1_flat = e1_yx[0] * width + e1_yx[1]
    a4_flat = a4_yx[0] * width + a4_yx[1]

    game.take_action("PUT", (0, e1_flat, a4_flat))

    # Now Player 2's turn with high iterations (like in crash: 2500)
    state = game.get_current_state()
    rust_search = hiivelabs_mcts.ZertzMCTS(rings=37, t=1)

    # Use parallel mode with high iterations to match crash scenario
    action_type, action_data = rust_search.search_parallel(
        state['spatial'].astype(np.float32),
        state['global'].astype(np.float32),
        # rings=37,
        iterations=2500,
        # t=1,
        # num_threads=16,
        verbose=True
    )

    print(f"\nAction returned: {action_type} {action_data}")

    # Verify the action is valid
    if action_type == "PUT":
        marble_type, dst_flat, rem_flat = action_data
        placement_mask, _ = game.get_valid_actions()

        # Check it's in the valid action mask
        dst_x = dst_flat % width
        dst_y = dst_flat // width

        assert placement_mask[marble_type, dst_flat, rem_flat] > 0, \
            f"Invalid placement: marble={marble_type}, dst={dst_flat}, remove={rem_flat}"

        print(f"Destination: ({dst_y}, {dst_x})")

        # Check it's a valid ring position
        assert game.board.state[0, dst_y, dst_x] == 1, \
            f"Destination ({dst_y}, {dst_x}) is not a valid ring"

        # This should not raise ValueError
        try:
            pos = game.board.position_from_yx((dst_y, dst_x))
            print(f"Position string: {pos}")
        except ValueError as e:
            pytest.fail(f"position_from_yx failed: {e}, coords=({dst_y}, {dst_x}), flat={dst_flat}")

        # Try to apply the action - this is where the crash occurred
        try:
            game.take_action(action_type, action_data)
            print("Action applied successfully!")
        except Exception as e:
            pytest.fail(f"Failed to apply action: {e}")

    print("\nTest passed - crash scenario no longer reproduces!")


def test_position_0_6_validity():
    """Check if position (0, 6) is valid on a 37-ring board."""
    game = ZertzGame(rings=37)
    width = game.board.config.width  # 7

    # Position (0, 6) from crash log
    y, x = 0, 6

    print(f"\nBoard width: {width}")
    print(f"Position ({y}, {x}) in bounds: {y < width and x < width}")
    print(f"Position ({y}, {x}) has ring: {game.board.state[0, y, x] == 1}")

    # Check if it's a valid ring position
    if game.board.state[0, y, x] == 1:
        try:
            pos_str = game.board.position_from_yx((y, x))
            print(f"Position ({y}, {x}) = {pos_str}")
        except ValueError as e:
            print(f"Position ({y}, {x}) raises ValueError: {e}")
    else:
        print(f"Position ({y}, {x}) is not a ring on the board")

    # Check what positions are valid rings
    print("\nValid ring positions on row 0:")
    valid_positions = []
    for x in range(width):
        if game.board.state[0, 0, x] == 1:
            try:
                pos_str = game.board.position_from_yx((0, x))
                valid_positions.append(f"(0,{x})={pos_str}")
            except ValueError:
                valid_positions.append(f"(0,{x})=ERROR")
    print(", ".join(valid_positions))