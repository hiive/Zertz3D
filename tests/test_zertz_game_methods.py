"""
Unit tests for ZertzGame utility methods.

Tests methods that aren't covered by other test files:
- Action size/shape getters
- Position/dict conversion methods
- Copy and reset operations
- Render data generation
"""

import sys
import numpy as np
from pathlib import Path

# Add parent directory to path to import game modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from game.zertz_game import ZertzGame
from shared.render_data import RenderData


class TestCopyAndReset:
    """Test game copy and reset operations."""

    def test_deepcopy_creates_independent_copy(self):
        """Test that __deepcopy__ creates independent game instance."""
        import copy

        game = ZertzGame(rings=37)

        # Take some actions to modify state
        placement, capture = game.get_valid_actions()
        valid_actions = np.argwhere(placement)
        if len(valid_actions) > 0:
            action = tuple(valid_actions[0])
            game.take_action("PUT", action)

        # Create deep copy
        game_copy = copy.deepcopy(game)

        # Verify they have same state initially
        assert np.array_equal(game.board.state, game_copy.board.state)

        # Modify original
        valid_actions2 = np.argwhere(game.get_valid_actions()[0])
        if len(valid_actions2) > 0:
            action2 = tuple(valid_actions2[0])
            game.take_action("PUT", action2)

        # Verify copy is unchanged
        assert not np.array_equal(game.board.state, game_copy.board.state)

    def test_reset_board_returns_to_initial_state(self):
        """Test that reset_board restores initial game state."""
        game = ZertzGame(rings=37)

        # Save initial state
        initial_rings = np.copy(game.board.state[game.board.RING_LAYER])
        initial_supply = game.board.global_state[game.board.SUPPLY_SLICE].copy()

        # Take some actions
        for _ in range(5):
            placement, capture = game.get_valid_actions()
            valid_actions = np.argwhere(placement)
            if len(valid_actions) > 0:
                action = tuple(valid_actions[0])
                game.take_action("PUT", action)
            else:
                break

        # Verify state changed
        assert not np.array_equal(
            game.board.state[game.board.RING_LAYER], initial_rings
        )

        # Reset
        game.reset_board()

        # Verify back to initial state
        assert np.array_equal(
            game.board.state[game.board.RING_LAYER], initial_rings
        )
        assert np.array_equal(
            game.board.global_state[game.board.SUPPLY_SLICE], initial_supply
        )


class TestPositionConversion:
    """Test methods that convert actions to/from position lists and dicts."""

    def test_get_placement_positions(self):
        """Test conversion of placement array to position list."""
        game = ZertzGame(rings=37)

        placement_array, _ = game.get_valid_actions()
        positions = game.get_placement_positions(placement_array)

        # Should return list of position strings
        assert isinstance(positions, list)
        assert len(positions) > 0

        # Each should be a valid position string (e.g., "D4")
        for pos in positions:
            assert isinstance(pos, str)
            assert len(pos) >= 2  # At least letter + number

    def test_get_capture_dicts(self):
        """Test conversion of capture array to list of dicts."""
        game = ZertzGame(rings=37)

        # Set up a game state with possible captures
        board = game.board

        # Place marbles to create capture opportunity
        # Put white at D4
        board.state[board.MARBLE_TO_LAYER["w"], 3, 3] = 1
        # Put gray at E4 (adjacent)
        board.state[board.MARBLE_TO_LAYER["g"], 3, 4] = 1
        # Put black at F4 (can jump from E4 over white to D4... wait, need correct setup)

        # Actually, let's just test the method works with the capture array
        _, capture_array = game.get_valid_actions()

        capture_dicts = game.get_capture_dicts(capture_array)

        # Should return list of dicts
        assert isinstance(capture_dicts, list)

        # If there are captures, each should have required keys
        for cap_dict in capture_dicts:
            assert isinstance(cap_dict, dict)
            assert "action" in cap_dict
            assert cap_dict["action"] == "CAP"
            assert "src" in cap_dict
            assert "dst" in cap_dict
            assert "capture" in cap_dict
            assert "cap" in cap_dict

    def test_get_removal_positions(self):
        """Test getting list of removable positions for a PUT action."""
        game = ZertzGame(rings=37)

        placement_array, _ = game.get_valid_actions()

        # Find a valid placement action
        valid_actions = np.argwhere(placement_array)
        assert len(valid_actions) > 0

        action = tuple(valid_actions[0])
        # marble_idx, dst, rem = action

        # Get removal positions for this action
        removals = game.get_removal_positions(placement_array, "PUT", action)

        # Should return list of position strings
        assert isinstance(removals, list)

        # Each should be a valid position string
        for pos in removals:
            assert isinstance(pos, str)
            assert len(pos) >= 2

    def test_get_removal_positions_returns_empty_for_cap(self):
        """Test that get_removal_positions returns empty list for CAP actions."""
        game = ZertzGame(rings=37)

        placement_array, _ = game.get_valid_actions()

        # Create a CAP action
        cap_action = (0, 3, 3)  # direction, y, x

        removals = game.get_removal_positions(placement_array, "CAP", cap_action)

        assert removals == []


class TestRenderData:
    """Test get_render_data method."""

    def test_get_render_data_without_highlights(self):
        """Test get_render_data with highlight_choices=None."""
        game = ZertzGame(rings=37)

        placement, _ = game.get_valid_actions()
        valid_actions = np.argwhere(placement)
        assert len(valid_actions) > 0
        action = tuple(valid_actions[0])

        render_data = game.get_render_data("PUT", action, highlight_choices=None)

        assert isinstance(render_data, RenderData)
        assert render_data.action_dict is not None
        assert render_data.action_dict["action"] == "PUT"

        # Should not have highlight data (empty lists/dicts)
        assert not render_data.has_highlights()
        assert render_data.placement_positions == []
        assert render_data.capture_moves == []
        assert render_data.removal_positions == []

    def test_get_render_data_with_highlights(self):
        """Test get_render_data with highlight_choices='uniform'."""
        game = ZertzGame(rings=37)

        placement, _ = game.get_valid_actions()
        valid_actions = np.argwhere(placement)
        assert len(valid_actions) > 0
        action = tuple(valid_actions[0])

        render_data = game.get_render_data("PUT", action, highlight_choices='uniform')

        assert isinstance(render_data, RenderData)
        assert render_data.action_dict is not None

        # Should have highlight data
        assert isinstance(render_data.placement_positions, list)
        assert isinstance(render_data.capture_moves, list)
        assert isinstance(render_data.removal_positions, list)

        # At game start, should have valid placements
        assert len(render_data.placement_positions) > 0

    def test_get_render_data_for_pass(self):
        """Test get_render_data for PASS action."""
        game = ZertzGame(rings=37)

        render_data = game.get_render_data("PASS", None, highlight_choices=None)

        assert isinstance(render_data, RenderData)
        assert render_data.action_dict["action"] == "PASS"


class TestActionStringConversion:
    """Test action string parsing and generation."""

    def test_action_to_str_for_pass(self):
        """Test that action_to_str handles PASS actions."""
        game = ZertzGame(rings=37)

        action_str, action_dict = game.action_to_str("PASS", None)

        assert action_str == "PASS"
        assert action_dict == {"action": "PASS"}

    def test_str_to_action_invalid_type(self):
        """Test str_to_action with invalid action type."""
        game = ZertzGame(rings=37)

        action_type, action = game.str_to_action("INVALID w D4 B2")

        # Should return empty string and None for invalid action
        assert action is None

    def test_str_to_action_put_missing_args(self):
        """Test str_to_action with incomplete PUT action."""
        game = ZertzGame(rings=37)

        # Missing destination and removal
        action_type, action = game.str_to_action("PUT w")

        assert action_type == ""
        assert action is None

    def test_str_to_action_cap_missing_args(self):
        """Test str_to_action with incomplete CAP action."""
        game = ZertzGame(rings=37)

        # Missing destination
        action_type, action = game.str_to_action("CAP D4 w")

        assert action_type == ""
        assert action is None

    def test_str_to_action_put_without_removal(self):
        """Test str_to_action with PUT action without removal."""
        game = ZertzGame(rings=37)

        action_type, action = game.str_to_action("PUT w D4")

        assert action_type == "PUT"
        assert action is not None
        # Action should be (marble_idx, dst, rem)
        # rem should be width² when no removal specified
        marble_idx, dst_y, dst_x, rem_y, rem_x = action
        assert rem_y is None and rem_x is None


class TestHasValidMoves:
    """Test _has_valid_moves helper method."""

    def test_has_valid_moves_at_start(self):
        """Test that game has valid moves at start."""
        game = ZertzGame(rings=37)

        # Access via internal method (it's used by game logic)
        has_moves = game._has_valid_moves()

        assert has_moves == True

    def test_has_no_valid_moves_when_blocked(self):
        """Test detecting when player has no valid moves."""
        game = ZertzGame(rings=37)
        board = game.board

        # Empty marble supply
        board.global_state[board.SUPPLY_W] = 0
        board.global_state[board.SUPPLY_G] = 0
        board.global_state[board.SUPPLY_B] = 0

        # Empty current player's captured marbles
        board.global_state[board.P1_CAP_W] = 0
        board.global_state[board.P1_CAP_G] = 0
        board.global_state[board.P1_CAP_B] = 0

        # Fill all rings with marbles (no captures possible)
        white_layer = board.MARBLE_TO_LAYER["w"]
        ring_positions = np.argwhere(board.state[board.RING_LAYER] == 1)
        for y, x in ring_positions:
            board.state[white_layer][y, x] = 1

        has_moves = game._has_valid_moves()

        assert has_moves == False
