"""
Unit tests for Notation Generation.

Tests the core notation generation functionality from action dictionaries
and ActionResult objects. Uses real game objects with minimal mocking.
"""

import pytest
import sys
from pathlib import Path
import numpy as np
from hiivelabs_mcts import algebraic_to_coordinate

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from game.zertz_game import ZertzGame
from game.zertz_board import ZertzBoard
from game.action_result import ActionResult


class TestNotationGeneration:
    """Test notation generation for various action types."""

    @pytest.mark.parametrize("rings", [
        ZertzBoard.SMALL_BOARD_37,
        ZertzBoard.MEDIUM_BOARD_48,
        ZertzBoard.LARGE_BOARD_61
    ])
    def test_placement_notation_without_removal(self, rings):
        """Test notation for placement without ring removal (all board sizes)."""
        game = ZertzGame(rings=rings)

        action_dict = {"action": "PUT", "marble": "w", "dst": "D4", "remove": ""}
        action_result = ActionResult(captured_marbles=None)

        notation = game.action_to_notation(action_dict, action_result)

        assert notation == "Wd4", f"Expected 'Wd4', got '{notation}'"

    @pytest.mark.parametrize("rings", [
        ZertzBoard.SMALL_BOARD_37,
        ZertzBoard.MEDIUM_BOARD_48,
        ZertzBoard.LARGE_BOARD_61
    ])
    def test_placement_notation_with_removal(self, rings):
        """Test notation for placement with ring removal (all board sizes)."""
        game = ZertzGame(rings=rings)

        action_dict = {"action": "PUT", "marble": "g", "dst": "D4", "remove": "B2"}
        action_result = ActionResult(captured_marbles=None)

        notation = game.action_to_notation(action_dict, action_result)

        assert notation == "Gd4,b2", f"Expected 'Gd4,b2', got '{notation}'"

    @pytest.mark.parametrize("rings", [
        ZertzBoard.SMALL_BOARD_37,
        ZertzBoard.MEDIUM_BOARD_48,
        ZertzBoard.LARGE_BOARD_61
    ])
    def test_placement_notation_with_isolation(self, rings):
        """Test notation for placement that causes isolation (all board sizes)."""
        game = ZertzGame(rings=rings)

        action_dict = {"action": "PUT", "marble": "b", "dst": "D4", "remove": "C3"}

        # Simulate isolation result
        action_result = ActionResult(
            captured_marbles=[
                {"marble": "w", "pos": "A1"},
                {"marble": "g", "pos": "B2"},
            ]
        )

        notation = game.action_to_notation(action_dict, action_result)

        # Should include isolation marker
        assert " x " in notation, f"Expected isolation marker in '{notation}'"
        assert notation.startswith("Bd4,c3 x "), (
            f"Expected 'Bd4,c3 x ...' prefix, got '{notation}'"
        )
        assert "Wa1" in notation, "Expected 'Wa1' in isolation captures"
        assert "Gb2" in notation, "Expected 'Gb2' in isolation captures"

    @pytest.mark.parametrize("rings", [
        ZertzBoard.SMALL_BOARD_37,
        ZertzBoard.MEDIUM_BOARD_48,
        ZertzBoard.LARGE_BOARD_61
    ])
    def test_capture_notation(self, rings):
        """Test notation for capture actions (all board sizes)."""
        game = ZertzGame(rings=rings)

        action_dict = {
            "action": "CAP",
            "marble": "w",
            "src": "C3",
            "dst": "E5",
            "capture": "g",
            "cap": "D4",
        }
        action_result = ActionResult(captured_marbles="g")

        notation = game.action_to_notation(action_dict, action_result)

        assert notation == "x c3Ge5", f"Expected 'x c3Ge5', got '{notation}'"

    @pytest.mark.parametrize("rings", [
        ZertzBoard.SMALL_BOARD_37,
        ZertzBoard.MEDIUM_BOARD_48,
        ZertzBoard.LARGE_BOARD_61
    ])
    def test_pass_notation(self, rings):
        """Test notation for pass actions (all board sizes)."""
        game = ZertzGame(rings=rings)

        action_dict = {"action": "PASS"}
        action_result = ActionResult(captured_marbles=None)

        notation = game.action_to_notation(action_dict, action_result)

        assert notation == "-", f"Expected '-', got '{notation}'"


class TestNotationImmediacy:
    """Test that notation is generated immediately, not deferred."""

    @pytest.mark.parametrize("rings", [
        ZertzBoard.SMALL_BOARD_37,
        ZertzBoard.MEDIUM_BOARD_48,
        ZertzBoard.LARGE_BOARD_61
    ])
    def test_notation_generated_with_action_result(self, rings):
        """Test that action_to_notation receives ActionResult and generates correct isolation notation (all board sizes)."""
        game = ZertzGame(rings=rings)

        action_dict = {"action": "PUT", "marble": "w", "dst": "D4", "remove": "C3"}
        action_result = ActionResult(captured_marbles=[{"marble": "w", "pos": "A1"}])

        # Should be able to generate notation immediately with result
        notation = game.action_to_notation(action_dict, action_result)

        # Notation should be exactly correct for this isolation
        expected = "Wd4,c3 x Wa1"
        assert notation == expected, f"Expected '{expected}', got '{notation}'"

    @pytest.mark.parametrize("rings", [
        ZertzBoard.SMALL_BOARD_37,
        ZertzBoard.MEDIUM_BOARD_48,
        ZertzBoard.LARGE_BOARD_61
    ])
    def test_no_buffering_required(self, rings):
        """Test that notation doesn't require buffering or deferred generation (all board sizes)."""
        game = ZertzGame(rings=rings)

        # Test multiple actions in sequence with exact expected output
        actions_and_expected = [
            (
                {"action": "PUT", "marble": "w", "dst": "D4", "remove": ""},
                ActionResult(None),
                "Wd4",
            ),
            (
                {"action": "PUT", "marble": "g", "dst": "E5", "remove": "C3"},
                ActionResult(None),
                "Ge5,c3",
            ),
            (
                {
                    "action": "CAP",
                    "marble": "w",
                    "src": "B2",
                    "dst": "D4",
                    "capture": "g",
                    "cap": "C3",
                },
                ActionResult("g"),
                "x b2Gd4",
            ),
        ]

        # Each action should generate correct notation immediately
        for action_dict, action_result, expected in actions_and_expected:
            notation = game.action_to_notation(action_dict, action_result)
            assert notation == expected, f"Expected '{expected}', got '{notation}'"


class TestNotationWorkflow:
    """Test notation generation in realistic game scenarios."""

    @pytest.mark.parametrize("rings", [
        ZertzBoard.SMALL_BOARD_37,
        ZertzBoard.MEDIUM_BOARD_48,
        ZertzBoard.LARGE_BOARD_61
    ])
    def test_full_placement_action_workflow(self, rings):
        """Test complete workflow: action execution → notation generation (all board sizes)."""
        game = ZertzGame(rings=rings)
        board = game.board

        # Use specific placement: place white marble at D4, remove C1
        d4_idx = algebraic_to_coordinate("D4", board.config)
        c1_idx = algebraic_to_coordinate("C1", board.config)
        d4_y, d4_x = d4_idx
        c1_y, c1_x = c1_idx
        d4_flat = d4_y * board.config.width + d4_x
        c1_flat = c1_y * board.config.width + c1_x

        marble_idx = 0  # White marble
        action = (marble_idx, d4_flat, c1_flat)

        # Generate action string before execution
        action_str, action_dict = game.action_to_str("PUT", action)

        # Verify action dict has expected values
        assert action_dict["action"] == "PUT"
        assert action_dict["marble"] == "w"
        assert action_dict["dst"] == "D4"
        assert action_dict["remove"] == "C1"

        # Execute action
        action_result = game.take_action("PUT", action)

        # Generate notation with result
        notation = game.action_to_notation(action_dict, action_result)

        # Verify notation is exactly correct
        assert notation == "Wd4,c1", f"Expected 'Wd4,c1', got '{notation}'"

    @pytest.mark.parametrize("rings", [
        ZertzBoard.SMALL_BOARD_37,
        ZertzBoard.MEDIUM_BOARD_48,
        ZertzBoard.LARGE_BOARD_61
    ])
    def test_isolation_workflow_end_to_end(self, rings):
        """Test isolation detection and notation generation (all board sizes)."""
        game = ZertzGame(rings=rings)
        board = game.board

        # Clear board and set up isolation scenario
        board.state[board.RING_LAYER] = 0
        board.state[board.MARBLE_LAYERS] = 0

        # Create minimal topology:
        # Main region: D4, E4 (2 rings)
        # Isolated region: C1 (1 ring with white marble)
        # Connection: D3 (will be removed to isolate C1)

        d4_idx = algebraic_to_coordinate("D4", board.config)
        e4_idx = algebraic_to_coordinate("E4", board.config)
        d3_idx = algebraic_to_coordinate("D3", board.config)
        c1_idx = algebraic_to_coordinate("C1", board.config)

        board.state[board.RING_LAYER][d4_idx] = 1
        board.state[board.RING_LAYER][e4_idx] = 1
        board.state[board.RING_LAYER][d3_idx] = 1
        board.state[board.RING_LAYER][c1_idx] = 1

        # Place white marble on C1
        white_layer = board.MARBLE_TO_LAYER["w"]
        board.state[white_layer][c1_idx] = 1

        # Prepare action: place marble at E4, remove D3 (isolates C1)
        e4_y, e4_x = e4_idx
        d3_y, d3_x = d3_idx
        e4_flat = e4_y * board.config.width + e4_x
        d3_flat = d3_y * board.config.width + d3_x

        action = (0, e4_flat, d3_flat)  # Use white marble
        action_str, action_dict = game.action_to_str("PUT", action)

        # Execute action
        action_result = game.take_action("PUT", action)

        # Verify isolation occurred
        assert action_result.is_isolation(), "Should have detected isolation"
        assert len(action_result.captured_marbles) > 0, (
            "Should have captured isolated marbles"
        )

        # Generate notation
        notation = game.action_to_notation(action_dict, action_result)

        # Verify notation includes isolation marker
        assert " x " in notation, (
            f"Notation should include isolation marker, got '{notation}'"
        )

    @pytest.mark.parametrize("rings", [
        ZertzBoard.SMALL_BOARD_37,
        ZertzBoard.MEDIUM_BOARD_48,
        ZertzBoard.LARGE_BOARD_61
    ])
    def test_capture_action_workflow(self, rings):
        """Test capture action execution and notation (all board sizes)."""
        game = ZertzGame(rings=rings)
        board = game.board

        # Set up simple capture scenario
        board.state[board.MARBLE_LAYERS] = 0

        # B2 (white) → C3 (gray) → D4 (empty)
        b2_idx = algebraic_to_coordinate("B2", board.config)
        c3_idx = algebraic_to_coordinate("C3", board.config)
        d4_idx = algebraic_to_coordinate("D4", board.config)

        white_layer = board.MARBLE_TO_LAYER["w"]
        gray_layer = board.MARBLE_TO_LAYER["g"]

        board.state[white_layer][b2_idx] = 1
        board.state[gray_layer][c3_idx] = 1

        # Find capture direction
        b2_y, b2_x = b2_idx
        c3_y, c3_x = c3_idx

        for dir_idx, (dy, dx) in enumerate(board.DIRECTIONS):
            if (b2_y + dy, b2_x + dx) == (c3_y, c3_x):
                # Convert from capture mask indices to action format using helper
                action = ZertzBoard.capture_indices_to_action(
                    dir_idx, b2_y, b2_x, board.config.width, board.DIRECTIONS
                )
                break

        # Execute capture
        action_str, action_dict = game.action_to_str("CAP", action)
        action_result = game.take_action("CAP", action)

        # Verify capture occurred
        assert action_result.captured_marbles == "g", "Should have captured gray marble"

        # Generate notation
        notation = game.action_to_notation(action_dict, action_result)

        # Verify notation format
        assert notation.startswith("x "), (
            f"Capture notation should start with 'x ', got '{notation}'"
        )
        assert "G" in notation, f"Should include captured gray marble, got '{notation}'"

    @pytest.mark.parametrize("rings", [
        ZertzBoard.SMALL_BOARD_37,
        ZertzBoard.MEDIUM_BOARD_48,
        ZertzBoard.LARGE_BOARD_61
    ])
    def test_multiple_captures_in_sequence(self, rings):
        """Test notation generation for chain capture (multiple captures in one turn) (all board sizes)."""
        game = ZertzGame(rings=rings)
        board = game.board

        # Set up chain capture scenario
        board.state[board.MARBLE_LAYERS] = 0

        # Create chain along same row: B2 (w) → C3 (g) → D4 (empty) → E4 (b) → F4 (empty)
        # Row A1 B2 C3 D4 E4 F4 G4 - all aligned horizontally
        b2_idx = algebraic_to_coordinate("B2", board.config)
        c3_idx = algebraic_to_coordinate("C3", board.config)
        d4_idx = algebraic_to_coordinate("D4", board.config)
        e4_idx = algebraic_to_coordinate("E4", board.config)

        white_layer = board.MARBLE_TO_LAYER["w"]
        gray_layer = board.MARBLE_TO_LAYER["g"]
        black_layer = board.MARBLE_TO_LAYER["b"]

        board.state[white_layer][b2_idx] = 1
        board.state[gray_layer][c3_idx] = 1
        board.state[black_layer][e4_idx] = 1

        # First capture: B2 → C3 → D4
        b2_y, b2_x = b2_idx
        c3_y, c3_x = c3_idx

        direction1 = None
        for dir_idx, (dy, dx) in enumerate(board.DIRECTIONS):
            if (b2_y + dy, b2_x + dx) == (c3_y, c3_x):
                direction1 = dir_idx
                # Convert from capture mask indices to action format using helper
                action1 = ZertzBoard.capture_indices_to_action(
                    dir_idx, b2_y, b2_x, board.config.width, board.DIRECTIONS
                )
                break

        assert direction1 is not None, "Should find direction from B2 to C3"

        action_str1, action_dict1 = game.action_to_str("CAP", action1)
        action_result1 = game.take_action("CAP", action1)
        notation1 = game.action_to_notation(action_dict1, action_result1)

        # Verify first notation is exactly correct
        assert notation1 == "x b2Gd4", f"Expected 'x b2Gd4', got '{notation1}'"
        assert action_result1.captured_marbles == "g", "Should capture gray marble"

        # Verify chain capture can continue: D4 → E4 → F4
        # Game should automatically require the chain capture
        placement_mask, capture_mask = game.get_valid_actions()
        d4_y, d4_x = d4_idx
        e4_y, e4_x = e4_idx

        # Capture should be forced from D4 (no placement moves allowed)
        assert not np.any(placement_mask), (
            "No placement moves during forced chain capture"
        )

        direction2 = None
        for dir_idx, (dy, dx) in enumerate(board.DIRECTIONS):
            if (d4_y + dy, d4_x + dx) == (e4_y, e4_x):
                # Check that this capture is available
                if capture_mask[dir_idx, d4_y, d4_x]:
                    direction2 = dir_idx
                    # Convert from capture mask indices to action format using helper
                    action2 = ZertzBoard.capture_indices_to_action(
                        dir_idx, d4_y, d4_x, board.config.width, board.DIRECTIONS
                    )
                    break

        assert direction2 is not None, (
            "Should find direction from D4 to E4 for chain capture"
        )

        action_str2, action_dict2 = game.action_to_str("CAP", action2)
        action_result2 = game.take_action("CAP", action2)
        notation2 = game.action_to_notation(action_dict2, action_result2)

        # Verify second notation is exactly correct
        # Note: hexagonal geometry differs across board sizes
        expected_notation2 = "x d4Bf3" if rings == 61 else "x d4Bf4"
        assert notation2 == expected_notation2, f"Expected '{expected_notation2}', got '{notation2}'"
        assert action_result2.captured_marbles == "b", "Should capture black marble"

    @pytest.mark.parametrize("rings", [
        ZertzBoard.SMALL_BOARD_37,
        ZertzBoard.MEDIUM_BOARD_48,
        ZertzBoard.LARGE_BOARD_61
    ])
    def test_pass_action_workflow(self, rings):
        """Test pass action when player has no valid moves (all board sizes)."""
        game = ZertzGame(rings=rings)
        board = game.board

        # Empty the marble pool and player's captured marbles
        board.global_state[board.SUPPLY_W] = 0
        board.global_state[board.SUPPLY_G] = 0
        board.global_state[board.SUPPLY_B] = 0
        board.global_state[board.P1_CAP_W] = 0
        board.global_state[board.P1_CAP_G] = 0
        board.global_state[board.P1_CAP_B] = 0

        # Fill all rings to prevent placement
        white_layer = board.MARBLE_TO_LAYER["w"]
        ring_positions = np.argwhere(board.state[board.RING_LAYER] == 1)
        for y, x in ring_positions:
            board.state[white_layer][y, x] = 1

        # Player should have no valid moves
        placement_mask, capture_mask = game.get_valid_actions()
        assert not np.any(placement_mask), "Should have no valid placements"
        assert not np.any(capture_mask), "Should have no valid captures"

        # Execute pass
        action_result = game.take_action("PASS", None)
        action_dict = {"action": "PASS"}
        notation = game.action_to_notation(action_dict, action_result)

        # Verify pass notation
        assert notation == "-", f"Pass notation should be '-', got '{notation}'"
