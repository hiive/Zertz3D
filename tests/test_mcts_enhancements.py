"""Tests for MCTS enhancements: Virtual Loss and FPU (First Play Urgency).

This test suite verifies Python-Rust parity for advanced MCTS features.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from learner.mcts.mcts_node import MCTSNode, VIRTUAL_LOSS, VIRTUAL_LOSS_VALUE
from learner.mcts.mcts_tree import MCTSTree
from game.zertz_game import ZertzGame
from game.zertz_board import ZertzBoard


class TestVirtualLoss:
    """Test Virtual Loss functionality.

    Note: Virtual Loss is reserved for Python 3.13+ free-threaded mode.
    These tests verify the API is present and works correctly even though
    it's not currently used in production (multiprocessing uses separate memory).
    """

    def test_virtual_loss_constants(self):
        """Verify Virtual Loss constants match Rust implementation."""
        # Rust: VIRTUAL_LOSS = 3, VIRTUAL_LOSS_SCALED = -3000
        # Python: VIRTUAL_LOSS = 3, VIRTUAL_LOSS_VALUE = -3.0
        assert VIRTUAL_LOSS == 3
        assert VIRTUAL_LOSS_VALUE == -3.0

    def test_virtual_loss_adds_and_removes_correctly(self):
        """Test Virtual Loss can be added and removed without affecting node."""
        game = ZertzGame(rings=ZertzBoard.SMALL_BOARD_37)
        config = game.board._get_config()

        node = MCTSNode(
            game.board.state,
            game.board.global_state,
            config,
            game.board.canonicalizer
        )

        # Initial state
        assert node.visits == 0
        assert node.value == 0

        # Add virtual loss
        node.add_virtual_loss()
        assert node.visits == VIRTUAL_LOSS
        assert node.value == VIRTUAL_LOSS_VALUE

        # Remove virtual loss
        node.remove_virtual_loss()
        assert node.visits == 0
        assert node.value == 0

    def test_virtual_loss_with_real_updates(self):
        """Test Virtual Loss interacts correctly with real updates."""
        game = ZertzGame(rings=ZertzBoard.SMALL_BOARD_37)
        config = game.board._get_config()

        node = MCTSNode(
            game.board.state,
            game.board.global_state,
            config,
            game.board.canonicalizer
        )

        # Add virtual loss
        node.add_virtual_loss()
        assert node.visits == VIRTUAL_LOSS
        assert node.value == VIRTUAL_LOSS_VALUE

        # Add real update
        node.visits += 1
        node.value += 0.5
        assert node.visits == VIRTUAL_LOSS + 1
        assert abs(node.value - (VIRTUAL_LOSS_VALUE + 0.5)) < 1e-6

        # Remove virtual loss
        node.remove_virtual_loss()
        assert node.visits == 1
        assert abs(node.value - 0.5) < 1e-6

    def test_multiple_virtual_losses(self):
        """Test multiple Virtual Losses can be added (for multiple threads)."""
        game = ZertzGame(rings=ZertzBoard.SMALL_BOARD_37)
        config = game.board._get_config()

        node = MCTSNode(
            game.board.state,
            game.board.global_state,
            config,
            game.board.canonicalizer
        )

        # Add virtual loss twice (simulating two threads)
        node.add_virtual_loss()
        node.add_virtual_loss()
        assert node.visits == 2 * VIRTUAL_LOSS
        assert node.value == 2 * VIRTUAL_LOSS_VALUE

        # Remove both
        node.remove_virtual_loss()
        node.remove_virtual_loss()
        assert node.visits == 0
        assert node.value == 0


class TestFPU:
    """Test First Play Urgency (FPU) functionality."""

    def test_fpu_unvisited_node_with_reduction(self):
        """Test FPU estimation for unvisited nodes with reduction=0.2."""
        game = ZertzGame(rings=ZertzBoard.SMALL_BOARD_37)
        config = game.board._get_config()

        # Create parent with some visits and value
        parent = MCTSNode(
            game.board.state,
            game.board.global_state,
            config,
            game.board.canonicalizer
        )
        parent.visits = 10
        parent.value = 5.0  # Average value = 0.5

        # Create unvisited child
        child = MCTSNode(
            game.board.state.copy(),
            game.board.global_state.copy(),
            config,
            game.board.canonicalizer
        )

        # Calculate UCB1 score with FPU
        mcts = MCTSTree()
        score = mcts.ucb1_score(child, parent, exploration_constant=1.41, fpu_reduction=0.2)

        # Expected: -(parent_avg_value - fpu_reduction) + exploration
        # parent_avg_value = 5.0 / 10 = 0.5
        # estimated_q = -(0.5 - 0.2) = -0.3
        # exploration = 1.41 * sqrt(10) ≈ 4.459
        # score ≈ -0.3 + 4.459 = 4.159
        import math
        expected_q = -(0.5 - 0.2)
        expected_u = 1.41 * math.sqrt(10)
        expected = expected_q + expected_u

        assert abs(score - expected) < 1e-6, f"FPU score mismatch: got {score}, expected {expected}"

    def test_fpu_unvisited_node_without_reduction(self):
        """Test standard UCB1 for unvisited nodes (no FPU)."""
        game = ZertzGame(rings=ZertzBoard.SMALL_BOARD_37)
        config = game.board._get_config()

        parent = MCTSNode(
            game.board.state,
            game.board.global_state,
            config,
            game.board.canonicalizer
        )
        parent.visits = 10
        parent.value = 5.0

        child = MCTSNode(
            game.board.state.copy(),
            game.board.global_state.copy(),
            config,
            game.board.canonicalizer
        )

        # Calculate UCB1 score without FPU
        mcts = MCTSTree()
        score = mcts.ucb1_score(child, parent, exploration_constant=1.41, fpu_reduction=None)

        # Should return infinity for standard UCB1
        assert score == float('inf'), "Without FPU, unvisited nodes should have infinite score"

    def test_fpu_visited_node_ignores_fpu(self):
        """Test that FPU has no effect on visited nodes."""
        game = ZertzGame(rings=ZertzBoard.SMALL_BOARD_37)
        config = game.board._get_config()

        parent = MCTSNode(
            game.board.state,
            game.board.global_state,
            config,
            game.board.canonicalizer
        )
        parent.visits = 10
        parent.value = 5.0

        child = MCTSNode(
            game.board.state.copy(),
            game.board.global_state.copy(),
            config,
            game.board.canonicalizer
        )
        # Visit the child
        child.visits = 1
        child.value = 0.6

        mcts = MCTSTree()

        # Score with FPU
        score_with_fpu = mcts.ucb1_score(
            child, parent, exploration_constant=1.41, fpu_reduction=0.2
        )

        # Score without FPU
        score_without_fpu = mcts.ucb1_score(
            child, parent, exploration_constant=1.41, fpu_reduction=None
        )

        # Should be identical for visited nodes
        assert score_with_fpu == score_without_fpu, \
            "FPU should not affect visited nodes"

        # Verify it matches standard UCB1 formula
        import math
        expected_q = -(child.value / child.visits)  # -0.6
        expected_u = 1.41 * math.sqrt(math.log(parent.visits) / child.visits)
        expected = expected_q + expected_u

        assert abs(score_with_fpu - expected) < 1e-6, \
            "Visited node score should match standard UCB1"

    def test_fpu_negative_parent_value(self):
        """Test FPU with negative parent value (losing position)."""
        game = ZertzGame(rings=ZertzBoard.SMALL_BOARD_37)
        config = game.board._get_config()

        # Parent has negative average value (losing)
        parent = MCTSNode(
            game.board.state,
            game.board.global_state,
            config,
            game.board.canonicalizer
        )
        parent.visits = 10
        parent.value = -4.0  # Average value = -0.4

        child = MCTSNode(
            game.board.state.copy(),
            game.board.global_state.copy(),
            config,
            game.board.canonicalizer
        )

        mcts = MCTSTree()
        score = mcts.ucb1_score(child, parent, exploration_constant=1.41, fpu_reduction=0.2)

        # Expected: -(parent_avg_value - fpu_reduction) + exploration
        # parent_avg_value = -4.0 / 10 = -0.4
        # estimated_q = -(-0.4 - 0.2) = -(-0.6) = 0.6
        # exploration = 1.41 * sqrt(10) ≈ 4.459
        import math
        expected_q = -((-0.4) - 0.2)
        expected_u = 1.41 * math.sqrt(10)
        expected = expected_q + expected_u

        assert abs(score - expected) < 1e-6, "FPU with negative parent value"
        assert expected_q > 0.0, "Negative parent value should result in positive estimated Q for child"

    def test_fpu_integration_with_search(self):
        """Test FPU integration in actual MCTS search."""
        game = ZertzGame(rings=ZertzBoard.SMALL_BOARD_37)
        mcts = MCTSTree()

        # Run search with FPU enabled
        action_with_fpu = mcts.search(
            game,
            iterations=100,
            fpu_reduction=0.2,
            verbose=False
        )

        # Run search without FPU (standard UCB1)
        game2 = ZertzGame(rings=ZertzBoard.SMALL_BOARD_37)
        action_without_fpu = mcts.search(
            game2,
            iterations=100,
            fpu_reduction=None,
            verbose=False
        )

        # Both should return valid actions (may be same or different)
        assert action_with_fpu is not None
        assert action_without_fpu is not None
        assert action_with_fpu[0] in ["PUT", "CAP", "PASS"]
        assert action_without_fpu[0] in ["PUT", "CAP", "PASS"]


class TestParity:
    """Test Python-Rust parity for Virtual Loss and FPU."""

    def test_virtual_loss_constants_match_rust(self):
        """Verify Python Virtual Loss constants match Rust."""
        # Rust: VIRTUAL_LOSS = 3, VIRTUAL_LOSS_SCALED = -3000 (scaled by 1000)
        # Python: VIRTUAL_LOSS = 3, VIRTUAL_LOSS_VALUE = -3.0
        assert VIRTUAL_LOSS == 3
        assert abs(VIRTUAL_LOSS_VALUE * 1000 - (-3000)) < 1e-3

    def test_fpu_formula_matches_rust(self):
        """Verify Python FPU formula matches Rust implementation."""
        # Both should use: -(parent_value - fpu_reduction) + exploration
        game = ZertzGame(rings=ZertzBoard.SMALL_BOARD_37)
        config = game.board._get_config()

        parent = MCTSNode(
            game.board.state,
            game.board.global_state,
            config,
            game.board.canonicalizer
        )
        parent.visits = 100
        parent.value = 25.0  # avg = 0.25

        child = MCTSNode(
            game.board.state.copy(),
            game.board.global_state.copy(),
            config,
            game.board.canonicalizer
        )

        mcts = MCTSTree()
        python_score = mcts.ucb1_score(
            child, parent, exploration_constant=1.41, fpu_reduction=0.2
        )

        # Calculate using Rust formula
        import math
        parent_avg = parent.value / parent.visits
        estimated_q = -(parent_avg - 0.2)
        exploration = 1.41 * math.sqrt(parent.visits)
        rust_score = estimated_q + exploration

        assert abs(python_score - rust_score) < 1e-6, \
            "Python and Rust FPU formulas should match"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])