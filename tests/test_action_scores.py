"""Tests for action scores functionality.

Verifies that players can return per-action scores that reflect move quality,
and that the scores are properly normalized to [0.0, 1.0] range.
"""

import hiivelabs_mcts.zertz as zertz

from game.zertz_game import ZertzGame
from game.players import MCTSZertzPlayer, RandomZertzPlayer


class TestActionScores:
    """Test action scores API across different player types."""

    def test_placement_with_no_removal(self):
        game = ZertzGame(rings=37)
        action = zertz.ZertzAction.placement(game.board.config, 0, 3, 4)
        action_type, action_data = action.to_tuple(game.board.config)
        assert action_type == 'PUT'
        assert len(action_data) == 5
        assert action_data[0] == 0
        assert action_data[1] == 3
        assert action_data[2] == 4
        assert action_data[3] is None
        assert action_data[4] is None


    def test_random_player_returns_uniform_scores(self):
        """RandomZertzPlayer should return uniform scores (all 1.0)."""
        game = ZertzGame(rings=37)
        player = RandomZertzPlayer(game, n=1)

        # Get action and scores
        action = player.get_action()
        scores = player.get_last_action_scores()

        # Verify action is in scores
        assert action in scores

        # All scores should be 1.0 (uniform)
        assert all(score == 1.0 for score in scores.values())

        # Should have at least one action
        assert len(scores) > 0

    def test_mcts_returns_normalized_scores(self):
        """MCTS should return normalized scores [0.0, 1.0]."""
        game = ZertzGame(rings=37)
        player = MCTSZertzPlayer(
            game,
            n=1,
            iterations=100,
            num_workers=1,
            rng_seed=42,
        )

        # Get action and scores
        action = player.get_action()
        scores = player.get_last_action_scores()

        # Verify action is in scores
        assert action in scores

        # All scores should be in [0.0, 1.0] range
        assert all(0.0 <= score <= 1.0 for score in scores.values())

        # Best action should have score of 1.0 (normalized to max)
        max_score = max(scores.values())
        assert max_score == 1.0

        # Selected action should have high score
        assert scores[action] >= 0.5  # Should be in top half

    def test_scores_match_action_format(self):
        """Action scores should use correct action tuple format."""
        game = ZertzGame(rings=37)
        player = MCTSZertzPlayer(
            game,
            n=1,
            iterations=50,
            num_workers=1,
            rng_seed=42,
        )

        # Get action and scores
        scores = player.get_last_action_scores()

        # Check that all keys are valid action tuples
        for action_key in scores.keys():

            # Verify action type
            action_type, action_data = action_key.to_tuple(game.board.config)
            assert action_type in ['PUT', 'CAP', 'PASS']
            # Verify action data format
            if action_type == 'PUT':
                assert action_data is not None
                assert len(action_data) == 5  # (marble_type, src_y, src_x, remove_y, remove_x)
            elif action_type == 'CAP':
                assert action_data is not None
                assert len(action_data) == 3  # (src_y, src_x, remove_y, remove_x)
            elif action_type == 'PASS':
                assert action_data is None

    def test_scores_count_matches_children(self):
        """Number of action scores should match number of children explored."""
        game = ZertzGame(rings=37)
        player = MCTSZertzPlayer(
            game,
            n=1,
            iterations=200,
            num_workers=1,
            rng_seed=42,
        )

        # Get action and scores
        player.get_action()
        scores = player.get_last_action_scores()

        # Number of scores should match last_root_children
        assert len(scores) == player._last_root_children
        assert player._last_root_children > 0

    def test_scores_reflect_visit_distribution(self):
        """Higher visit counts should produce higher scores."""
        game = ZertzGame(rings=37)
        player = MCTSZertzPlayer(
            game,
            n=1,
            iterations=500,  # More iterations for better distribution
            num_workers=1,
            rng_seed=42,
        )

        # Get action and scores
        action = player.get_action()
        scores = player.get_last_action_scores()

        # Should have multiple actions with varied scores
        assert len(scores) > 1

        # Should have a range of scores (not all the same)
        unique_scores = set(scores.values())
        assert len(unique_scores) > 1

        # Best action should have highest score
        best_action = max(scores.items(), key=lambda x: x[1])[0]
        assert scores[best_action] == 1.0

    def test_scores_available_after_search(self):
        """Scores should be available after MCTS search completes."""
        game = ZertzGame(rings=37)
        player = MCTSZertzPlayer(
            game,
            n=1,
            iterations=100,
            num_workers=1,
            rng_seed=42,
        )

        # Make some moves and verify scores are available after MCTS
        for _ in range(5):
            action = player.get_action()
            scores = player.get_last_action_scores()

            # Scores should always be available
            assert scores is not None
            assert len(scores) > 0

            # If action required MCTS (not forced), it should be in scores
            # Forced moves (single capture/placement) skip MCTS, so scores won't include them
            placement_mask, capture_mask = game.get_valid_actions()
            c_count = capture_mask.sum()
            p_count = placement_mask.sum()

            # Only check if the action was a choice (required MCTS)
            if c_count > 1 or (c_count == 0 and p_count > 1):
                assert action in scores

            # Apply action to game
            game.take_action(action)

            # Check if game ended
            if game.get_game_ended():
                break

    def test_parallel_returns_scores(self):
        """Parallel MCTS should also return action scores."""
        game = ZertzGame(rings=37)
        player = MCTSZertzPlayer(
            game,
            n=1,
            iterations=100,
            num_workers=4,
            rng_seed=42,
        )

        # Get action and scores
        action = player.get_action()
        scores = player.get_last_action_scores()

        # Verify basic properties
        assert action in scores
        assert all(0.0 <= score <= 1.0 for score in scores.values())
        assert max(scores.values()) == 1.0