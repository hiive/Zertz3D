import numpy as np
from hiivelabs_mcts import algebraic_to_coordinate

from game.zertz_game import ZertzGame
from game.zertz_board import ZertzBoard


def test_put_allows_no_removal_when_sentinel_used():
    """Placing a marble should succeed without ring removal when index==width²."""
    game = ZertzGame(rings=ZertzBoard.SMALL_BOARD_37)
    board = game.board

    # Ensure deterministic supply for the move (1 white left, others empty)
    board.global_state[board.SUPPLY_W] = 1
    board.global_state[board.SUPPLY_G] = 0
    board.global_state[board.SUPPLY_B] = 0

    dest = algebraic_to_coordinate("D4", board.config)
    put_loc = (dest[0], dest[1])
    no_removal = (None, None)

    rings_before = int(np.sum(board.state[board.RING_LAYER]))

    game.take_action("PUT", (0, *put_loc, *no_removal))

    # Marble placed
    white_layer = board.MARBLE_TO_LAYER["w"]
    assert board.state[white_layer, dest[0], dest[1]] == 1

    # Ring count unchanged (no removal occurred)
    assert int(np.sum(board.state[board.RING_LAYER])) == rings_before


def test_no_removable_rings_after_placement():
    """Test placement when no edge rings can be removed (all occupied)."""
    game = ZertzGame(rings=ZertzBoard.SMALL_BOARD_37)
    board = game.board

    # Create scenario where all edge rings are occupied
    # Place marbles on all edge rings except one in the center
    white_layer = board.MARBLE_TO_LAYER["w"]

    # Find all edge rings
    edge_rings = []
    all_rings = np.argwhere(board.state[board.RING_LAYER] == 1)
    for y, x in all_rings:
        neighbors = board.get_neighbors((y, x))
        # Edge rings have at least one missing neighbor
        has_missing_neighbor = False
        for ny, nx in neighbors:
            if not board._is_inbounds((ny, nx)) or board.state[board.RING_LAYER, ny, nx] == 0:
                has_missing_neighbor = True
                break
        if has_missing_neighbor:
            edge_rings.append((y, x))

    # Place marbles on all edge rings
    for y, x in edge_rings:
        board.state[white_layer, y, x] = 1

    # Find an open center position
    center_pos = algebraic_to_coordinate("D4", board.config)
    assert board.state[white_layer, center_pos[0], center_pos[1]] == 0, "Center should be open"

    # Configure supply
    board.global_state[board.SUPPLY_W] = 1
    board.global_state[board.SUPPLY_G] = 0
    board.global_state[board.SUPPLY_B] = 0

    # Get current player before action
    player_before = int(board.global_state[board.CUR_PLAYER])
    rings_before = int(np.sum(board.state[board.RING_LAYER]))

    # Place marble at center position with no removal (width² sentinel)
    put_loc = (center_pos[0], center_pos[1])
    no_removal = (None, None)

    # Get the action dict that would be generated
    _, action_dict = game.action_to_str("PUT", (0, *put_loc, *no_removal))

    game.take_action("PUT", (0, *put_loc, *no_removal))

    # Verify marble was placed
    assert board.state[white_layer, center_pos[0], center_pos[1]] == 1, "Marble should be placed"

    # Verify no ring was removed
    rings_after = int(np.sum(board.state[board.RING_LAYER]))
    assert rings_after == rings_before, "No ring should be removed"

    # Verify turn passed to next player
    player_after = int(board.global_state[board.CUR_PLAYER])
    assert player_after == (player_before + 1) % 2, "Turn should pass to next player"

    # Verify notation is correct (no comma when no ring removed)
    notation = game.action_to_notation(action_dict, action_result=None)
    assert notation == "Wd4", f"Expected 'Wd4', got '{notation}'"
    assert "," not in notation, "Notation should not contain comma when no ring removed"


def test_no_removable_rings_notation():
    """Test that notation is correct when no ring is removed."""
    game = ZertzGame(rings=ZertzBoard.SMALL_BOARD_37)

    # Test action dict with empty remove field
    action_dict = {"action": "PUT", "marble": "w", "dst": "D4", "remove": ""}
    notation = game.action_to_notation(action_dict, action_result=None)

    # Should be just the marble color and position, no comma
    assert notation == "Wd4", f"Expected 'Wd4', got '{notation}'"
    assert "," not in notation, "Should not contain comma when no ring removed"


def test_board_full_last_move_awards_previous_player():
    """When the board is filled, the player who moved last should win."""
    game = ZertzGame(rings=ZertzBoard.SMALL_BOARD_37)
    board = game.board

    # Fill every ring except one with white marbles
    white_layer = board.MARBLE_TO_LAYER["w"]
    all_rings = np.argwhere(board.state[board.RING_LAYER] == 1)
    last_ring = tuple(all_rings[-1])

    for y, x in all_rings[:-1]:
        board.state[white_layer, y, x] = 1

    # Configure supply so Player 1 can place the final marble
    board.global_state[board.SUPPLY_W] = 1
    board.global_state[board.SUPPLY_G] = 0
    board.global_state[board.SUPPLY_B] = 0
    # Give Player 1 one captured marble to avoid BOTH_LOSE condition
    board.global_state[board.P1_CAP_SLICE] = [1, 0, 0]
    board.global_state[board.P2_CAP_SLICE] = 0
    board.global_state[board.CUR_PLAYER] = board.PLAYER_1

    put_loc = (last_ring[0], last_ring[1])
    no_removal = (None, None)

    # Player 1 makes the final placement (no removable rings remain)
    game.take_action("PUT", (0, *put_loc, *no_removal))

    assert np.all(np.sum(board.state[board.BOARD_LAYERS], axis=0) != 1)
    assert game.get_game_end_reason() == "Board completely filled with marbles"
    assert game.get_game_ended() == 1
