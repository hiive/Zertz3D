"""
Unit tests for get_game_end_reason() method.

Tests that the game provides detailed human-readable reasons for all game end conditions.
"""

import pytest
import sys
from pathlib import Path
import numpy as np

# Add parent directory to path to import game modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from game.constants import BLITZ_MARBLES, BLITZ_WIN_CONDITIONS
from game.zertz_game import ZertzGame
from game.zertz_board import ZertzBoard


class TestGameEndReasons:
    """Test detailed game end reason messages."""

    @pytest.mark.parametrize("rings", [
        ZertzBoard.SMALL_BOARD_37,
        ZertzBoard.MEDIUM_BOARD_48,
        ZertzBoard.LARGE_BOARD_61
    ])
    def test_win_reason_4_white(self, rings):
        """Test reason message for 4 white marbles win (all board sizes)."""
        game = ZertzGame(rings=rings)
        board = game.board

        # Set up win condition: 4 whites
        board.global_state[board.P1_CAP_W] = 4
        board.global_state[board.P1_CAP_G] = 2
        board.global_state[board.P1_CAP_B] = 1

        reason = game.get_game_end_reason()
        assert reason == "Captured required marbles: 4 white"

    @pytest.mark.parametrize("rings", [
        ZertzBoard.SMALL_BOARD_37,
        ZertzBoard.MEDIUM_BOARD_48,
        ZertzBoard.LARGE_BOARD_61
    ])
    def test_win_reason_5_gray(self, rings):
        """Test reason message for 5 gray marbles win (all board sizes)."""
        game = ZertzGame(rings=rings)
        board = game.board

        # Set up win condition: 5 grays
        board.global_state[board.P2_CAP_W] = 1
        board.global_state[board.P2_CAP_G] = 5
        board.global_state[board.P2_CAP_B] = 2

        reason = game.get_game_end_reason()
        assert reason == "Captured required marbles: 5 gray"

    @pytest.mark.parametrize("rings", [
        ZertzBoard.SMALL_BOARD_37,
        ZertzBoard.MEDIUM_BOARD_48,
        ZertzBoard.LARGE_BOARD_61
    ])
    def test_win_reason_6_black(self, rings):
        """Test reason message for 6 black marbles win (all board sizes)."""
        game = ZertzGame(rings=rings)
        board = game.board

        # Set up win condition: 6 blacks
        board.global_state[board.P1_CAP_W] = 2
        board.global_state[board.P1_CAP_G] = 1
        board.global_state[board.P1_CAP_B] = 6

        reason = game.get_game_end_reason()
        assert reason == "Captured required marbles: 6 black"

    @pytest.mark.parametrize("rings", [
        ZertzBoard.SMALL_BOARD_37,
        ZertzBoard.MEDIUM_BOARD_48,
        ZertzBoard.LARGE_BOARD_61
    ])
    def test_win_reason_3_of_each(self, rings):
        """Test reason message for 3 of each color win (all board sizes)."""
        game = ZertzGame(rings=rings)
        board = game.board

        # Set up win condition: 3 of each
        board.global_state[board.P1_CAP_W] = 3
        board.global_state[board.P1_CAP_G] = 3
        board.global_state[board.P1_CAP_B] = 3

        reason = game.get_game_end_reason()
        assert reason == "Captured required marbles: 3 of each color"

    def test_blitz_win_reason_3_white(self):
        """Test reason message for 3 white in blitz mode."""
        game = ZertzGame(
            rings=ZertzBoard.SMALL_BOARD_37,
            marbles=BLITZ_MARBLES,
            win_con=BLITZ_WIN_CONDITIONS,
        )
        board = game.board

        # Set up blitz win condition: 3 whites
        board.global_state[board.P1_CAP_W] = 3
        board.global_state[board.P1_CAP_G] = 1
        board.global_state[board.P1_CAP_B] = 0

        reason = game.get_game_end_reason()
        assert reason == "Captured required marbles: 3 white"

    def test_blitz_win_reason_2_of_each(self):
        """Test reason message for 2 of each in blitz mode."""
        game = ZertzGame(
            rings=ZertzBoard.SMALL_BOARD_37,
            marbles=BLITZ_MARBLES,
            win_con=BLITZ_WIN_CONDITIONS,
        )
        board = game.board

        # Set up blitz win condition: 2 of each
        board.global_state[board.P2_CAP_W] = 2
        board.global_state[board.P2_CAP_G] = 2
        board.global_state[board.P2_CAP_B] = 2

        reason = game.get_game_end_reason()
        assert reason == "Captured required marbles: 2 of each color"

    @pytest.mark.parametrize("rings", [
        ZertzBoard.SMALL_BOARD_37,
        ZertzBoard.MEDIUM_BOARD_48,
        ZertzBoard.LARGE_BOARD_61
    ])
    def test_board_full_reason(self, rings):
        """Test reason message for board completely filled (all board sizes)."""
        game = ZertzGame(rings=rings)
        board = game.board

        # Set up non-winning captured marbles
        board.global_state[board.P1_CAP_W] = 1
        board.global_state[board.P1_CAP_G] = 1
        board.global_state[board.P1_CAP_B] = 1

        # Fill entire board with marbles
        white_layer = board.MARBLE_TO_LAYER["w"]
        ring_positions = np.argwhere(board.state[board.RING_LAYER] == 1)
        for y, x in ring_positions:
            board.state[white_layer][y, x] = 1

        reason = game.get_game_end_reason()
        assert reason == "Board completely filled with marbles"

    @pytest.mark.parametrize("rings", [
        ZertzBoard.SMALL_BOARD_37,
        ZertzBoard.MEDIUM_BOARD_48,
        ZertzBoard.LARGE_BOARD_61
    ])
    def test_no_marbles_reason(self, rings):
        """Test reason message for opponent having no marbles (all board sizes)."""
        game = ZertzGame(rings=rings)
        board = game.board

        # Empty the marble pool
        board.global_state[board.SUPPLY_W] = 0
        board.global_state[board.SUPPLY_G] = 0
        board.global_state[board.SUPPLY_B] = 0

        # Empty current player's (P1) captured marbles
        board.global_state[board.P1_CAP_W] = 0
        board.global_state[board.P1_CAP_G] = 0
        board.global_state[board.P1_CAP_B] = 0

        # Set current player
        board.global_state[board.CUR_PLAYER] = board.PLAYER_1

        reason = game.get_game_end_reason()
        assert reason == "Opponent has no marbles left to place"

    @pytest.mark.parametrize("rings", [
        ZertzBoard.SMALL_BOARD_37,
        ZertzBoard.MEDIUM_BOARD_48,
        ZertzBoard.LARGE_BOARD_61
    ])
    def test_move_loop_reason(self, rings):
        """Test reason message for move loop (repeated position) (all board sizes)."""
        game = ZertzGame(rings=rings)

        # Create a repeating move pattern
        move_pattern = [
            ("PUT", (0, 10, 5)),
            ("PUT", (1, 15, 8)),
            ("PUT", (0, 12, 7)),
            ("PUT", (1, 18, 11)),
        ]
        game.move_history = move_pattern + move_pattern

        reason = game.get_game_end_reason()
        assert reason == "Move loop detected (repeated position)"

    @pytest.mark.parametrize("rings", [
        ZertzBoard.SMALL_BOARD_37,
        ZertzBoard.MEDIUM_BOARD_48,
        ZertzBoard.LARGE_BOARD_61
    ])
    def test_immobilization_with_winner_reason(self, rings):
        """Test reason message for both players immobilized with Player 1 winner (all board sizes)."""
        game = ZertzGame(rings=rings)
        board = game.board

        # Empty the marble pool
        board.global_state[board.SUPPLY_W] = 0
        board.global_state[board.SUPPLY_G] = 0
        board.global_state[board.SUPPLY_B] = 0

        # Set up win condition for P1: 4 whites
        board.global_state[board.P1_CAP_W] = 4
        board.global_state[board.P1_CAP_G] = 2
        board.global_state[board.P1_CAP_B] = 1

        # P2 doesn't have win condition
        board.global_state[board.P2_CAP_W] = 2
        board.global_state[board.P2_CAP_G] = 2
        board.global_state[board.P2_CAP_B] = 2

        # Fill all rings with marbles
        white_layer = board.MARBLE_TO_LAYER["w"]
        ring_positions = np.argwhere(board.state[board.RING_LAYER] == 1)
        for y, x in ring_positions:
            board.state[white_layer][y, x] = 1

        # Both players pass
        board.global_state[board.CUR_PLAYER] = board.PLAYER_1
        game.take_action("PASS", None)
        game.take_action("PASS", None)

        reason = game.get_game_end_reason()
        assert reason == "Both players immobilized: 4 white"

    @pytest.mark.parametrize("rings", [
        ZertzBoard.SMALL_BOARD_37,
        ZertzBoard.MEDIUM_BOARD_48,
        ZertzBoard.LARGE_BOARD_61
    ])
    def test_immobilization_player2_winner_reason(self, rings):
        """Test reason message for both players immobilized with Player 2 winner (all board sizes)."""
        game = ZertzGame(rings=rings)
        board = game.board

        # Empty the marble pool
        board.global_state[board.SUPPLY_W] = 0
        board.global_state[board.SUPPLY_G] = 0
        board.global_state[board.SUPPLY_B] = 0

        # Set up win condition for P2: 5 grays
        board.global_state[board.P2_CAP_W] = 2
        board.global_state[board.P2_CAP_G] = 5
        board.global_state[board.P2_CAP_B] = 1

        # P1 doesn't have win condition
        board.global_state[board.P1_CAP_W] = 2
        board.global_state[board.P1_CAP_G] = 2
        board.global_state[board.P1_CAP_B] = 2

        # Fill all rings with marbles
        white_layer = board.MARBLE_TO_LAYER["w"]
        ring_positions = np.argwhere(board.state[board.RING_LAYER] == 1)
        for y, x in ring_positions:
            board.state[white_layer][y, x] = 1

        # Both players pass
        board.global_state[board.CUR_PLAYER] = board.PLAYER_1
        game.take_action("PASS", None)
        game.take_action("PASS", None)

        reason = game.get_game_end_reason()
        assert reason == "Both players immobilized: 5 gray"

    @pytest.mark.parametrize("rings", [
        ZertzBoard.SMALL_BOARD_37,
        ZertzBoard.MEDIUM_BOARD_48,
        ZertzBoard.LARGE_BOARD_61
    ])
    def test_immobilization_tie_reason(self, rings):
        """Test reason message for both players immobilized with no winner (all board sizes)."""
        game = ZertzGame(rings=rings)
        board = game.board

        # Empty the marble pool
        board.global_state[board.SUPPLY_W] = 0
        board.global_state[board.SUPPLY_G] = 0
        board.global_state[board.SUPPLY_B] = 0

        # Set up equal non-winning captured marbles
        board.global_state[board.P1_CAP_W] = 2
        board.global_state[board.P1_CAP_G] = 2
        board.global_state[board.P1_CAP_B] = 2
        board.global_state[board.P2_CAP_W] = 2
        board.global_state[board.P2_CAP_G] = 2
        board.global_state[board.P2_CAP_B] = 2

        # Fill all rings with marbles
        white_layer = board.MARBLE_TO_LAYER["w"]
        ring_positions = np.argwhere(board.state[board.RING_LAYER] == 1)
        for y, x in ring_positions:
            board.state[white_layer][y, x] = 1

        # Both players pass
        board.global_state[board.CUR_PLAYER] = board.PLAYER_1
        game.take_action("PASS", None)
        game.take_action("PASS", None)

        reason = game.get_game_end_reason()
        assert reason == "Both players immobilized with no winner"

    @pytest.mark.parametrize("rings", [
        ZertzBoard.SMALL_BOARD_37,
        ZertzBoard.MEDIUM_BOARD_48,
        ZertzBoard.LARGE_BOARD_61
    ])
    def test_no_reason_when_game_not_over(self, rings):
        """Test that None is returned when game is not over (all board sizes)."""
        game = ZertzGame(rings=rings)

        reason = game.get_game_end_reason()
        assert reason is None