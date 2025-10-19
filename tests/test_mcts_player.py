"""Tests for MCTS player implementation."""

import pytest
from game.zertz_game import ZertzGame
from game.players.mcts_zertz_player import MCTSZertzPlayer
from game.zertz_player import RandomZertzPlayer


class TestMCTSZertzPlayer:
    """Test suite for MCTSZertzPlayer."""

    def test_initialization(self):
        """Test that MCTS player can be initialized."""
        game = ZertzGame(rings=37)
        player = MCTSZertzPlayer(game, n=1, iterations=10, verbose=False)

        assert player.game == game
        assert player.n == 1
        assert player.iterations == 10
        assert player.captured == {"w": 0, "g": 0, "b": 0}

    def test_get_action_returns_valid_format(self):
        """Test that get_action returns correct format."""
        game = ZertzGame(rings=37)
        player = MCTSZertzPlayer(game, n=1, iterations=10, verbose=False)

        action = player.get_action()

        # Should return tuple of (action_type, action_data)
        assert isinstance(action, tuple)
        assert len(action) == 2
        action_type, action_data = action

        # Action type should be PUT, CAP, or PASS
        assert action_type in ["PUT", "CAP", "PASS"]

        # First move should be PUT (no captures available)
        assert action_type == "PUT"
        assert isinstance(action_data, tuple)
        assert len(action_data) == 3

    def test_same_interface_as_random_player(self):
        """Test that MCTS player has same interface as RandomZertzPlayer."""
        game = ZertzGame(rings=37)
        random_player = RandomZertzPlayer(game, n=1)
        mcts_player = MCTSZertzPlayer(game, n=1, iterations=10, verbose=False)

        random_action = random_player.get_action()
        mcts_action = mcts_player.get_action()

        # Both should return same format
        assert type(random_action) == type(mcts_action)
        assert len(random_action) == len(mcts_action)

        # Both should have same action type on first move (PUT)
        assert random_action[0] == mcts_action[0] == "PUT"

    def test_mcts_with_minimal_iterations(self):
        """Test MCTS with very low iteration count."""
        game = ZertzGame(rings=37)
        player = MCTSZertzPlayer(game, n=1, iterations=1, verbose=False)

        action = player.get_action()
        assert action[0] in ["PUT", "CAP", "PASS"]

    def test_mcts_verbose_mode(self):
        """Test that verbose mode doesn't crash."""
        game = ZertzGame(rings=37)
        player = MCTSZertzPlayer(game, n=1, iterations=5, verbose=True)

        # Should print statistics without crashing
        action = player.get_action()
        assert action is not None

    def test_transposition_table_optional(self):
        """Test that transposition table can be disabled."""
        game = ZertzGame(rings=37)
        player = MCTSZertzPlayer(
            game, n=1, iterations=10, use_transposition_table=False, verbose=False
        )

        assert player.transposition_table is None
        action = player.get_action()
        assert action[0] in ["PUT", "CAP", "PASS"]

    def test_transposition_lookups_configurable(self):
        """Test that transposition lookups can be disabled."""
        game = ZertzGame(rings=37)

        # With lookups enabled
        player1 = MCTSZertzPlayer(
            game,
            n=1,
            iterations=10,
            use_transposition_table=True,
            use_transposition_lookups=True,
            verbose=False,
        )
        action1 = player1.get_action()

        # With lookups disabled
        game2 = ZertzGame(rings=37)
        player2 = MCTSZertzPlayer(
            game2,
            n=1,
            iterations=10,
            use_transposition_table=True,
            use_transposition_lookups=False,
            verbose=False,
        )
        action2 = player2.get_action()

        # Both should return valid actions
        assert action1[0] in ["PUT", "CAP", "PASS"]
        assert action2[0] in ["PUT", "CAP", "PASS"]

    def test_mcts_can_play_full_game(self):
        """Test that MCTS player can play multiple moves."""
        game = ZertzGame(rings=37)
        player1 = MCTSZertzPlayer(game, n=1, iterations=3, verbose=False)
        player2 = MCTSZertzPlayer(game, n=2, iterations=3, verbose=False)

        move_count = 0
        max_moves = 10  # Just test a few moves, not a full game

        while game.get_game_ended() is None and move_count < max_moves:
            # Get current player
            current_player = player1 if game.board.get_cur_player() == 0 else player2

            # Get and apply action
            action_type, action_data = current_player.get_action()

            if action_type == "PASS":
                game.take_action("PASS", None)
            else:
                game.take_action(action_type, action_data)

            move_count += 1

        # Game should end (either win or tie)
        assert game.get_game_ended() is not None or move_count >= max_moves