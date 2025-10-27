"""Tests for blitz mode with MCTS."""

import pytest
import numpy as np
from game.zertz_game import ZertzGame
from game.constants import BLITZ_WIN_CONDITIONS, BLITZ_MARBLES, STANDARD_WIN_CONDITIONS
from game.players.mcts_zertz_player import MCTSZertzPlayer


class TestBlitzModeConstants:
    """Test blitz mode constants are correctly defined."""

    def test_blitz_win_conditions_values(self):
        """Verify blitz win condition thresholds are correct."""
        assert BLITZ_WIN_CONDITIONS == [
            {"w": 2, "g": 2, "b": 2},  # 2 of each (vs 3 in standard)
            {"w": 3},                   # 3 white (vs 4 in standard)
            {"g": 4},                   # 4 gray (vs 5 in standard)
            {"b": 5}                    # 5 black (vs 6 in standard)
        ]

    def test_blitz_marble_supply(self):
        """Verify blitz marble supply is correct."""
        assert BLITZ_MARBLES == {
            "w": 5,  # vs 6 in standard
            "g": 7,  # vs 8 in standard
            "b": 9   # vs 10 in standard
        }


class TestBlitzModeMCTS:
    """Test suite for blitz mode with MCTS."""

    def test_blitz_mode_detection(self):
        """Test that blitz mode is correctly detected."""
        game = ZertzGame(
            rings=37,
            win_con=BLITZ_WIN_CONDITIONS,
            marbles=BLITZ_MARBLES,
            t=1
        )

        player = MCTSZertzPlayer(
            game=game,
            n=1,
            iterations=100,
            num_workers=1,
            verbose=False
        )

        assert player._is_blitz_mode() is True
        assert game.win_con == BLITZ_WIN_CONDITIONS

    def test_standard_mode_detection(self):
        """Test that standard mode is correctly detected."""
        game = ZertzGame(rings=37, t=1)

        player = MCTSZertzPlayer(
            game=game,
            n=1,
            iterations=100,
            num_workers=1,
            verbose=False
        )

        assert player._is_blitz_mode() is False
        assert game.win_con == STANDARD_WIN_CONDITIONS

    def test_blitz_mode_mcts_search(self):
        """Test that MCTS search works correctly with blitz mode."""
        np.random.seed(12345)
        game = ZertzGame(
            rings=37,
            win_con=BLITZ_WIN_CONDITIONS,
            marbles=BLITZ_MARBLES,
            t=1
        )

        player = MCTSZertzPlayer(
            game=game,
            n=1,
            iterations=100,
            num_workers=1,
            verbose=False
        )

        # Should successfully return a valid action
        action = player.get_action()

        assert isinstance(action, tuple)
        assert len(action) == 2
        action_type, action_data = action
        assert action_type in ["PUT", "CAP", "PASS"]

        # First move should be PUT (no captures available)
        assert action_type == "PUT"
        assert isinstance(action_data, tuple)
        assert len(action_data) == 3

    def test_standard_mode_mcts_search(self):
        """Test that MCTS search works correctly with standard mode."""
        np.random.seed(12345)
        game = ZertzGame(rings=37, t=1)

        player = MCTSZertzPlayer(
            game=game,
            n=1,
            iterations=100,
            num_workers=1,
            verbose=False
        )

        # Should successfully return a valid action
        action = player.get_action()

        assert isinstance(action, tuple)
        assert len(action) == 2
        action_type, action_data = action
        assert action_type in ["PUT", "CAP", "PASS"]

        # First move should be PUT
        assert action_type == "PUT"
        assert isinstance(action_data, tuple)
        assert len(action_data) == 3

    def test_blitz_mode_parallel_search(self):
        """Test that parallel MCTS search works with blitz mode."""
        np.random.seed(54321)
        game = ZertzGame(
            rings=37,
            win_con=BLITZ_WIN_CONDITIONS,
            marbles=BLITZ_MARBLES,
            t=1
        )

        player = MCTSZertzPlayer(
            game=game,
            n=1,
            iterations=100,
            num_workers=4,
            verbose=False
        )

        # Should successfully return a valid action
        action = player.get_action()

        assert isinstance(action, tuple)
        assert len(action) == 2
        action_type, _ = action
        assert action_type in ["PUT", "CAP", "PASS"]

    def test_mode_passed_to_search(self):
        """Test that blitz parameter is correctly passed to search."""
        game_blitz = ZertzGame(
            rings=37,
            win_con=BLITZ_WIN_CONDITIONS,
            marbles=BLITZ_MARBLES,
            t=1
        )
        game_standard = ZertzGame(rings=37, t=1)

        player_blitz = MCTSZertzPlayer(
            game=game_blitz,
            n=1,
            verbose=False
        )
        player_standard = MCTSZertzPlayer(
            game=game_standard,
            n=1,
            verbose=False
        )

        # Verify mode detection is different
        assert player_blitz._is_blitz_mode() is True
        assert player_standard._is_blitz_mode() is False

        # Verify both can get current state
        state_blitz = game_blitz.get_current_state()
        state_standard = game_standard.get_current_state()

        assert state_blitz is not None
        assert state_standard is not None