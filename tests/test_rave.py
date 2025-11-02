"""Tests for RAVE (Rapid Action Value Estimation) in MCTS."""

import numpy as np
import pytest

from hiivelabs_mcts import ZertzMCTS
from game.zertz_game import ZertzGame
from game.constants import BLITZ_MARBLES, BLITZ_WIN_CONDITIONS


class TestRAVEBasics:
    """Test basic RAVE functionality and configuration."""

    def test_rave_disabled_by_default(self):
        """RAVE should be disabled when rave_constant is not provided."""
        mcts = ZertzMCTS(rings=37, )
        # Just verify construction works - RAVE should be None internally
        assert mcts is not None

    def test_rave_enabled_with_constant(self):
        """RAVE should be enabled when rave_constant is provided."""
        mcts = ZertzMCTS(rings=37, rave_constant=1000.0)
        assert mcts is not None

    def test_rave_constant_values(self):
        """Test various rave_constant values."""
        # Conservative
        mcts1 = ZertzMCTS(rings=37, rave_constant=300.0)
        assert mcts1 is not None

        # Balanced
        mcts2 = ZertzMCTS(rings=37, rave_constant=1000.0)
        assert mcts2 is not None

        # Aggressive
        mcts3 = ZertzMCTS(rings=37, rave_constant=3000.0)
        assert mcts3 is not None

    def test_rave_with_other_parameters(self):
        """RAVE should work with other MCTS parameters."""
        mcts = ZertzMCTS(rings=37, 
            exploration_constant=1.41,
            fpu_reduction=0.25,
            rave_constant=1000.0,
            use_transposition_table=True,
        )
        assert mcts is not None


class TestRAVESearch:
    """Test RAVE in actual MCTS searches."""

    def test_rave_search_completes(self):
        """RAVE-enabled search should complete without errors."""
        game = ZertzGame(rings=37)
        mcts = ZertzMCTS(rings=37, rave_constant=1000.0)

        state_dict = game.get_current_state()
        action_str, _ = mcts.search(
            spatial_state=state_dict['spatial'],
            global_state=state_dict['global'],
            # rings=37,
            iterations=100,
            seed=42,
        )

        assert action_str in ["PUT", "CAP", "PASS"]

    def test_rave_parallel_search_completes(self):
        """RAVE-enabled parallel search should complete without errors."""
        game = ZertzGame(rings=37)
        mcts = ZertzMCTS(rings=37, rave_constant=1000.0)

        state_dict = game.get_current_state()
        action_str, _ = mcts.search_parallel(
            spatial_state=state_dict['spatial'],
            global_state=state_dict['global'],
            # rings=37,
            iterations=100,
            seed=42,
            # num_threads=4,
        )

        assert action_str in ["PUT", "CAP", "PASS"]

    def test_rave_deterministic_with_seed(self):
        """RAVE searches should be deterministic with same seed."""
        game = ZertzGame(rings=37)
        mcts1 = ZertzMCTS(rings=37, rave_constant=1000.0)
        mcts2 = ZertzMCTS(rings=37, rave_constant=1000.0)

        state_dict = game.get_current_state()

        action1, data1 = mcts1.search(
            spatial_state=state_dict['spatial'],
            global_state=state_dict['global'],
            # rings=37,
            iterations=100,
            seed=12345,
        )

        action2, data2 = mcts2.search(
            spatial_state=state_dict['spatial'],
            global_state=state_dict['global'],
            # rings=37,
            iterations=100,
            seed=12345,
        )

        assert action1 == action2
        assert data1 == data2


class TestRAVEComparison:
    """Compare RAVE-enabled vs standard MCTS behavior."""

    def test_rave_produces_valid_actions(self):
        """RAVE should produce valid actions like standard MCTS."""
        game = ZertzGame(rings=37)

        spatial, global_state, _ = game.get_current_state().values()

        # Standard MCTS
        mcts_standard = ZertzMCTS(rings=37, )
        action_standard, _ = mcts_standard.search(
            spatial_state=spatial,
            global_state=global_state,
            # rings=37,
            iterations=100,
            seed=42,
        )

        # RAVE MCTS
        mcts_rave = ZertzMCTS(rings=37, rave_constant=1000.0)
        action_rave, _ = mcts_rave.search(
            spatial_state=spatial,
            global_state=global_state,
            # rings=37,
            iterations=100,
            seed=42,
        )

        # Both should produce valid action types
        assert action_standard in ["PUT", "CAP", "PASS"]
        assert action_rave in ["PUT", "CAP", "PASS"]

    def test_rave_statistics_available(self):
        """RAVE search should populate child statistics."""
        game = ZertzGame(rings=37)
        mcts = ZertzMCTS(rings=37, rave_constant=1000.0)

        spatial, global_state, _ = game.get_current_state().values()
        mcts.search(
            spatial_state=spatial,
            global_state=global_state,
            # rings=37,
            iterations=200,
            seed=42,
        )

        # Check that statistics were collected
        assert mcts.last_root_visits() > 0
        assert mcts.last_root_children() > 0

        # Get child statistics
        stats = mcts.last_child_statistics()
        assert len(stats) > 0

        # Each stat should be (action_type, action_data, score)
        for action_type, action_data, score in stats:
            assert action_type in ["PUT", "CAP", "PASS"]
            assert 0.0 <= score <= 1.0


class TestRAVEEdgeCases:
    """Test RAVE behavior in edge cases."""

    def test_rave_with_few_iterations(self):
        """RAVE should work even with very few iterations."""
        game = ZertzGame(rings=37)
        mcts = ZertzMCTS(rings=37, rave_constant=1000.0)

        spatial, global_state, _ = game.get_current_state().values()
        action_str, _ = mcts.search(
            spatial_state=spatial,
            global_state=global_state,
            # rings=37,
            iterations=10,
            seed=42,
        )

        assert action_str in ["PUT", "CAP", "PASS"]

    def test_rave_with_many_iterations(self):
        """RAVE should handle many iterations efficiently."""
        game = ZertzGame(rings=37)
        mcts = ZertzMCTS(rings=37, rave_constant=1000.0)

        spatial, global_state, _ = game.get_current_state().values()
        action_str, _ = mcts.search(
            spatial_state=spatial,
            global_state=global_state,
            # rings=37,
            iterations=1000,
            seed=42,
        )

        assert action_str in ["PUT", "CAP", "PASS"]
        assert mcts.last_root_visits() >= 1000

    def test_rave_different_board_sizes(self):
        """RAVE should work with different board sizes."""
        for rings in [37, 48, 61]:
            game = ZertzGame(rings=rings)
            mcts = ZertzMCTS(rings=37, rave_constant=1000.0)

            spatial, global_state, _ = game.get_current_state().values()
            action_str, _ = mcts.search(
                spatial_state=spatial,
                global_state=global_state,
                # rings=rings,
                iterations=100,
                seed=42,
            )

            assert action_str in ["PUT", "CAP", "PASS"]


class TestRAVEBlitzMode:
    """Test RAVE in Blitz mode."""

    def test_rave_blitz_mode(self):
        """RAVE should work in Blitz mode."""
        game = ZertzGame(rings=37, marbles=BLITZ_MARBLES, win_con=BLITZ_WIN_CONDITIONS)
        mcts = ZertzMCTS(rings=37, rave_constant=1000.0)

        spatial, global_state, _ = game.get_current_state().values()
        action_str, _ = mcts.search(
            spatial_state=spatial,
            global_state=global_state,
            # rings=37,
            iterations=100,
            seed=42,
            # blitz=True,
        )

        assert action_str in ["PUT", "CAP", "PASS"]