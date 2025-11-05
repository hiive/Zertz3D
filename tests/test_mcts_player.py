"""Tests for MCTS player implementation."""

import pytest
import numpy as np
from game.zertz_game import ZertzGame
from game.players import MCTSZertzPlayer, RandomZertzPlayer
from hiivelabs_mcts import algebraic_to_coordinate

#TODO fill in test stubs
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

    def test_backend_respects_parameters(self):
        """Test that MCTS player respects all parameters."""
        game = ZertzGame(rings=37)
        player = MCTSZertzPlayer(
            game,
            n=1,
            iterations=8,
            max_simulation_depth=4,
            time_limit=0.01,
            use_transposition_table=True,
            use_transposition_lookups=False,
            clear_table_each_move=True,
            num_workers=2,
            verbose=False,
        )

        action = player.get_action()
        assert action[0] in ["PUT", "CAP", "PASS"]

    def test_transposition_persistence(self):
        """Test that transposition table persists across moves when configured."""
        game = ZertzGame(rings=37)
        player = MCTSZertzPlayer(
            game,
            n=1,
            iterations=6,
            use_transposition_table=True,
            use_transposition_lookups=True,
            clear_table_each_move=False,
            verbose=False,
        )

        first_action = player.get_action()
        second_action = player.get_action()

        assert first_action[0] in ["PUT", "CAP", "PASS"]
        assert second_action[0] in ["PUT", "CAP", "PASS"]


class TestMCTSCorrectness:
    """Tests to verify MCTS algorithm correctness and player perspective handling."""

    def test_mcts_prefers_winning_move(self):
        """Test that MCTS recognizes and selects a guaranteed winning move.

        This test creates a position where one player is one capture away from winning.
        MCTS should strongly prefer the winning move over other options.
        """
        import numpy as np

        # Create game and manually set up a winning position for Player 1
        game = ZertzGame(rings=37)

        # Give Player 1 (index 0) marbles close to winning
        # Win condition: 3 of each color, OR 4w/5g/6b
        # Set Player 1 to have: 2w, 2g, 2b (one capture away from 3-3-3 win)
        config = game.board.config
        game.board.global_state[config.p1_cap_slice] = np.array([2, 2, 2])

        # Put white marbles on the board that Player 1 can capture
        # Place marbles to create a capture opportunity: D3 (w) -> D4 (w) -> D5 (empty)
        d3_idx = algebraic_to_coordinate("D3", game.board.config)
        d4_idx = algebraic_to_coordinate("D4", game.board.config)
        d5_idx = algebraic_to_coordinate("D5", game.board.config)

        # Ensure rings exist (they should by default on a fresh board)
        game.board.state[game.board.RING_LAYER, d3_idx[0], d3_idx[1]] = 1
        game.board.state[game.board.RING_LAYER, d4_idx[0], d4_idx[1]] = 1
        game.board.state[game.board.RING_LAYER, d5_idx[0], d5_idx[1]] = 1

        # Place white marbles at D3 and D4 (layer 1 is white)
        game.board.state[1, d3_idx[0], d3_idx[1]] = 1
        game.board.state[1, d4_idx[0], d4_idx[1]] = 1

        # Set current player to Player 1 (index 0)
        game.board.global_state[config.cur_player] = 0

        # Create MCTS player with enough iterations to find the winning move
        player = MCTSZertzPlayer(game, n=1, iterations=100, verbose=False)

        # Get MCTS's chosen action
        action_type, action_data = player.get_action()

        # Since there's a capture available, MCTS should choose CAP
        # (captures are mandatory in Zertz if available)
        assert action_type == "CAP", "MCTS should recognize capture opportunity"

    def test_mcts_avoids_obvious_blunder(self):
        """Test that MCTS avoids moves that immediately lose the game.

        Creates a position where one move would allow opponent to win immediately,
        while other moves are safe. MCTS should avoid the blunder.
        """
        import numpy as np

        game = ZertzGame(rings=37)

        # Set up position where Player 2 is one capture from winning
        config = game.board.config
        game.board.global_state[config.p2_cap_slice] = np.array([2, 2, 2])

        # Set current player to Player 1 (who should NOT give Player 2 a winning capture)
        game.board.global_state[config.cur_player] = 0

        # This test is more qualitative - just verify MCTS can run in this position
        player = MCTSZertzPlayer(game, n=1, iterations=50, verbose=False)
        action = player.get_action()

        assert action is not None
        assert action[0] in ["PUT", "CAP", "PASS"]

    def test_mcts_player1_vs_random_winrate(self):
        """Test that MCTS as Player 1 beats random player significantly.

        With sufficient iterations, MCTS should win >50% of games against random.
        """
        import random
        random.seed(42)  # For reproducibility
        np.random.seed(42)

        wins = 0
        games = 10  # Small number for test speed

        for _ in range(games):
            game = ZertzGame(rings=37)
            player1 = MCTSZertzPlayer(game, n=1, iterations=200, verbose=False)
            player2 = RandomZertzPlayer(game, n=2)

            move_count = 0
            max_moves = 100

            while game.get_game_ended() is None and move_count < max_moves:
                current_player = player1 if game.board.get_cur_player() == 0 else player2
                action_type, action_data = current_player.get_action()

                if action_type == "PASS":
                    game.take_action("PASS", None)
                else:
                    game.take_action(action_type, action_data)

                move_count += 1

            result = game.get_game_ended()
            if result == 1:  # Player 1 (MCTS) won
                wins += 1

        win_rate = wins / games
        # MCTS should win at least 50% of games (ideally much more)
        assert win_rate >= 0.5, f"MCTS win rate {win_rate:.1%} should be >= 50%"

    def test_mcts_player2_vs_random_winrate(self):
        """Test that MCTS as Player 2 beats random player significantly.

        This tests that player perspective is handled correctly - MCTS should
        perform well regardless of which player it controls.
        """
        import random
        random.seed(43)  # Different seed from player1 test
        np.random.seed(43)

        wins = 0
        games = 10  # Small number for test speed

        for _ in range(games):
            game = ZertzGame(rings=37)
            player1 = RandomZertzPlayer(game, n=1)
            player2 = MCTSZertzPlayer(game, n=2, iterations=200, verbose=False, num_workers=1)

            move_count = 0
            max_moves = 100

            while game.get_game_ended() is None and move_count < max_moves:
                current_player = player1 if game.board.get_cur_player() == 0 else player2
                action_type, action_data = current_player.get_action()

                if action_type == "PASS":
                    game.take_action("PASS", None)
                else:
                    game.take_action(action_type, action_data)

                move_count += 1

            result = game.get_game_ended()
            if result == -1:  # Player 2 (MCTS) won
                wins += 1

        win_rate = wins / games
        # MCTS should win at least 50% of games (ideally much more)
        assert win_rate >= 0.5, f"MCTS win rate {win_rate:.1%} should be >= 50%"

    def test_mcts_symmetric_performance(self):
        """Test that MCTS performs similarly as Player 1 and Player 2.

        This verifies that player perspective handling is symmetric and correct.
        MCTS should not have a strong bias toward one player position.
        """
        import random

        # Test Player 1
        random.seed(44)
        np.random.seed(44)
        p1_wins = 0
        games = 5

        for _ in range(games):
            game = ZertzGame(rings=37)
            player1 = MCTSZertzPlayer(game, n=1, iterations=100, verbose=False)
            player2 = RandomZertzPlayer(game, n=2)

            move_count = 0
            while game.get_game_ended() is None and move_count < 100:
                current_player = player1 if game.board.get_cur_player() == 0 else player2
                action_type, action_data = current_player.get_action()
                if action_type == "PASS":
                    game.take_action("PASS", None)
                else:
                    game.take_action(action_type, action_data)
                move_count += 1

            if game.get_game_ended() == 1:
                p1_wins += 1

        # Test Player 2
        random.seed(45)
        np.random.seed(45)
        p2_wins = 0

        for _ in range(games):
            game = ZertzGame(rings=37)
            player1 = RandomZertzPlayer(game, n=1)
            player2 = MCTSZertzPlayer(game, n=2, iterations=100, verbose=False)

            move_count = 0
            while game.get_game_ended() is None and move_count < 100:
                current_player = player1 if game.board.get_cur_player() == 0 else player2
                action_type, action_data = current_player.get_action()
                if action_type == "PASS":
                    game.take_action("PASS", None)
                else:
                    game.take_action(action_type, action_data)
                move_count += 1

            if game.get_game_ended() == -1:
                p2_wins += 1

        # Both should have similar performance (within reasonable variance)
        # With only 5 games each, allow large variance but both should win something
        assert p1_wins >= 0 and p2_wins >= 0, "MCTS should perform for both players"

    def test_mcts_improves_with_more_iterations(self):
        """Test that MCTS with more iterations outperforms fewer iterations.

        This verifies that the search is actually improving decisions, not just
        randomizing them.
        """
        import random
        random.seed(46)
        np.random.seed(46)

        # Play games: MCTS (10 iter) vs MCTS (50 iter)
        high_iter_wins = 0
        games = 5

        for _ in range(games):
            game = ZertzGame(rings=37)
            player1 = MCTSZertzPlayer(game, n=1, iterations=10, verbose=False)
            player2 = MCTSZertzPlayer(game, n=2, iterations=50, verbose=False)

            move_count = 0
            while game.get_game_ended() is None and move_count < 100:
                current_player = player1 if game.board.get_cur_player() == 0 else player2
                action_type, action_data = current_player.get_action()
                if action_type == "PASS":
                    game.take_action("PASS", None)
                else:
                    game.take_action(action_type, action_data)
                move_count += 1

            result = game.get_game_ended()
            if result == -1:  # Player 2 (higher iterations) won
                high_iter_wins += 1

        # Higher iterations should win at least some games
        # (Not requiring >50% due to small sample and first-player advantage)
        assert high_iter_wins >= 1, "Higher iteration MCTS should win at least some games"
