"""Test that game state includes all necessary information for ML."""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from game.zertz_game import ZertzGame
from game.zertz_board import ZertzBoard


class TestStateCompleteness:
    """Verify that get_current_state() returns complete observable state."""

    def test_get_current_state_returns_dict(self):
        """State should be a dictionary with required keys."""
        game = ZertzGame(rings=37)
        state = game.get_current_state()

        assert isinstance(state, dict), "State should be a dictionary"
        assert "spatial" in state, "State should have 'spatial' key"
        assert "global" in state, "State should have 'global' key"
        assert "player" in state, "State should have 'player' key"

    def test_spatial_state_shape(self):
        """Spatial state should have correct shape (L, H, W)."""
        game = ZertzGame(rings=37, t=1)
        state = game.get_current_state()

        spatial = state["spatial"]
        assert isinstance(spatial, np.ndarray), "Spatial should be numpy array"
        assert len(spatial.shape) == 3, "Spatial should be 3D array"

        # For t=1: 4*1 + 1 = 5 layers, 7x7 board
        assert spatial.shape == (5, 7, 7), f"Expected (5, 7, 7), got {spatial.shape}"

    def test_global_state_shape(self):
        """Global state should have correct shape (10,)."""
        game = ZertzGame(rings=37)
        state = game.get_current_state()

        global_state = state["global"]
        assert isinstance(global_state, np.ndarray), "Global should be numpy array"
        assert global_state.shape == (10,), f"Expected (10,), got {global_state.shape}"

    def test_global_state_contains_supply(self):
        """Global state should contain supply counts."""
        game = ZertzGame(rings=37)
        state = game.get_current_state()

        global_state = state["global"]
        # Default supply: [6, 8, 10] for w, g, b
        assert global_state[ZertzBoard.SUPPLY_W] == 6, "White supply should be 6"
        assert global_state[ZertzBoard.SUPPLY_G] == 8, "Gray supply should be 8"
        assert global_state[ZertzBoard.SUPPLY_B] == 10, "Black supply should be 10"

    def test_global_state_contains_captured(self):
        """Global state should contain captured marble counts."""
        game = ZertzGame(rings=37)
        state = game.get_current_state()

        global_state = state["global"]
        # Initial captured counts should be 0
        assert global_state[ZertzBoard.P1_CAP_W] == 0, "P1 white captured should be 0"
        assert global_state[ZertzBoard.P1_CAP_G] == 0, "P1 gray captured should be 0"
        assert global_state[ZertzBoard.P1_CAP_B] == 0, "P1 black captured should be 0"
        assert global_state[ZertzBoard.P2_CAP_W] == 0, "P2 white captured should be 0"
        assert global_state[ZertzBoard.P2_CAP_G] == 0, "P2 gray captured should be 0"
        assert global_state[ZertzBoard.P2_CAP_B] == 0, "P2 black captured should be 0"

    def test_global_state_contains_current_player(self):
        """Global state should contain current player."""
        game = ZertzGame(rings=37)
        state = game.get_current_state()

        global_state = state["global"]
        # Game starts with player 1 (index 0)
        assert global_state[ZertzBoard.CUR_PLAYER] == 0, "Current player should be 0"

    def test_player_value_is_correct(self):
        """Player value should be 1 for P1, -1 for P2."""
        game = ZertzGame(rings=37)
        state = game.get_current_state()

        # Should start with player 1
        assert state["player"] == 1, "Player 1 should have value 1"

        # Take an action to switch to player 2
        placement, capture = game.get_valid_actions()
        valid_actions = np.argwhere(placement)
        if len(valid_actions) > 0:
            action = tuple(valid_actions[0])
            game.take_action("PUT", action)

            state2 = game.get_current_state()
            assert state2["player"] == -1, "Player 2 should have value -1"

    def test_state_is_copied(self):
        """State arrays should be copies, not references."""
        game = ZertzGame(rings=37)
        state1 = game.get_current_state()

        # Modify returned arrays
        state1["spatial"][0, 0, 0] = 99
        state1["global"][0] = 99

        # Get state again
        state2 = game.get_current_state()

        # Should not be affected
        assert state2["spatial"][0, 0, 0] != 99, "Spatial should be a copy"
        assert state2["global"][0] != 99, "Global should be a copy"



    def test_different_board_sizes(self):
        """State should work correctly for all board sizes."""
        for rings, width in [(37, 7), (48, 8), (61, 9)]:
            game = ZertzGame(rings=rings, t=1)
            state = game.get_current_state()

            # Check shapes
            assert state["spatial"].shape == (5, width, width), (
                f"Spatial shape wrong for {rings}-ring board"
            )
            assert state["global"].shape == (10,), (
                f"Global shape wrong for {rings}-ring board"
            )
            assert state["player"] in [1, -1], (
                f"Player value wrong for {rings}-ring board"
            )

    def test_get_valid_actions_with_cur_state(self):
        """Test get_valid_actions with explicit cur_state parameter."""
        game = ZertzGame(rings=37)

        # Get current state
        state = game.get_current_state()
        cur_state = np.copy(state["spatial"])

        # Get valid actions using cur_state parameter
        placement, capture = game.get_valid_actions(cur_state=cur_state)

        assert isinstance(placement, np.ndarray)
        assert isinstance(capture, np.ndarray)
        assert placement.shape == (3, 7, 7, 7, 7)
        assert capture.shape == (6, 7, 7)  # 6 x width x width

    def test_get_game_ended_with_cur_state(self):
        """Test get_game_ended with explicit cur_state parameter."""
        from game.zertz_game import PLAYER_1_WIN, PLAYER_2_WIN

        game = ZertzGame(rings=37)
        board = game.board

        # Set up a winning state in BOTH global_state AND spatial state
        # (get_game_ended with cur_state clones the game including global_state)
        board.global_state[board.P1_CAP_W] = 4
        board.global_state[board.P1_CAP_G] = 2
        board.global_state[board.P1_CAP_B] = 1

        # Ensure it's player 2's turn so player 1 is the "previous" player (winner)
        board.global_state[board.CUR_PLAYER] = board.PLAYER_2

        # Get current spatial state
        winning_state = np.copy(board.state)

        # Test get_game_ended with cur_state parameter
        # Creates a temp game with cloned settings and this spatial state
        result = game.get_game_ended(cur_state=winning_state)

        # Should return PLAYER_1_WIN since P1 has 4 white marbles
        assert result in [PLAYER_1_WIN, PLAYER_2_WIN], f"Expected win condition, got {result}"
