import numpy as np

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

    dest = board.str_to_index("D4")
    put_loc = board._2d_to_flat(*dest)
    no_removal = board.width**2

    rings_before = int(np.sum(board.state[board.RING_LAYER]))

    game.take_action("PUT", (0, put_loc, no_removal))

    # Marble placed
    white_layer = board.MARBLE_TO_LAYER["w"]
    assert board.state[white_layer, dest[0], dest[1]] == 1

    # Ring count unchanged (no removal occurred)
    assert int(np.sum(board.state[board.RING_LAYER])) == rings_before


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
    board.global_state[board.P1_CAP_SLICE] = 0
    board.global_state[board.P2_CAP_SLICE] = 0
    board.global_state[board.CUR_PLAYER] = board.PLAYER_1

    put_loc = board._2d_to_flat(*last_ring)
    no_removal = board.width**2

    # Player 1 makes the final placement (no removable rings remain)
    game.take_action("PUT", (0, put_loc, no_removal))

    assert np.all(np.sum(board.state[board.BOARD_LAYERS], axis=0) != 1)
    assert game.get_game_end_reason() == "Board completely filled with marbles"
    assert game.get_game_ended() == 1
