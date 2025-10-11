"""
Integration tests for Temporal Coupling Issue 2: End-to-End Isolation Notation.

These tests verify the complete flow from action execution through isolation
detection to correct notation in the log file.
"""

import pytest
import sys
import tempfile
import os
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from controller.zertz_game_controller import ZertzGameController
from game.zertz_game import ZertzGame
from game.zertz_board import ZertzBoard
from game.zertz_player import ReplayZertzPlayer


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        original_dir = os.getcwd()
        os.chdir(tmpdir)
        yield tmpdir
        os.chdir(original_dir)


@pytest.fixture
def game():
    """Create a standard game."""
    return ZertzGame(rings=37)


# ============================================================================
# Helper Functions
# ============================================================================

def create_isolation_scenario(board):
    """Create a board state that will cause isolation on next move.

    Sets up G1 isolated with a marble, ready to be captured.
    Returns the action that will cause isolation.
    """
    # G1 is at bottom-right corner
    # Its neighbors are: G2 (above), F1 (left), F2 (diagonal upper-left)

    # Place marble on G1
    g1_idx = board.str_to_index("G1")
    board.state[board.MARBLE_LAYERS.start, *g1_idx] = 1  # white marble

    # Remove G2 and F1 to prepare for isolation
    board.state[board.RING_LAYER, *board.str_to_index("G2")] = 0
    board.state[board.RING_LAYER, *board.str_to_index("F1")] = 0

    # Action: Place marble on D4, remove F2 to isolate G1
    marble_type = 0  # white
    dst_flat = board._2d_to_flat(*board.str_to_index("D4"))
    remove_flat = board._2d_to_flat(*board.str_to_index("F2"))

    return (marble_type, dst_flat, remove_flat)


def read_notation_file(filename):
    """Read notation file and return list of moves."""
    with open(filename, 'r') as f:
        lines = f.readlines()
        # Skip first line (board size) and empty lines
        moves = [l.strip() for l in lines[1:] if l.strip()]
        return moves


# ============================================================================
# Basic Integration Tests
# ============================================================================

def test_isolation_notation_written_to_file(temp_dir):
    """Test that isolation notation is correctly written to log file."""
    # Create game with logging enabled
    controller = ZertzGameController(
        rings=37,
        headless=True,
        log_notation=True,
        seed=999
    )

    # Set up isolation scenario
    action = create_isolation_scenario(controller.session.game.board)

    # Create mock task
    class Task:
        delay_time = 0
        done = False
        again = True

    task = Task()

    # Create replay player with the isolation action
    from game.zertz_player import RandomZertzPlayer
    player1 = RandomZertzPlayer(controller.session.game, n=1)

    # Manually execute the action that causes isolation
    ax, ay = 'PUT', action

    # Get action dict
    _, action_dict = controller.session.game.action_to_str(ax, ay)

    # Log action
    controller.logger.log_action(player1.n, action_dict)

    # Generate and buffer initial notation
    notation = controller.session.game.action_to_notation(action_dict, None)
    controller.logger.buffer_notation(notation)

    # Execute action (should return isolation)
    action_result = controller.session.game.take_action(ax, ay)

    # If isolation occurred, update notation
    if action_result.is_isolation() and action_result.has_captures():
        notation_with_isolation = controller.session.game.action_to_notation(action_dict, action_result.captured_marbles)
        controller.logger.update_buffered_notation(notation_with_isolation)

    # Close logger to write buffered notation
    controller._close_log_file()

    # Verify notation file
    notation_file = f"zertzlog_999_notation.txt"
    assert os.path.exists(notation_file)

    moves = read_notation_file(notation_file)

    # Should have the move with isolation notation
    assert len(moves) > 0
    # Isolation notation should contain ' x '
    assert any(' x ' in move for move in moves)


def test_multiple_isolations_in_sequence(temp_dir):
    """Test multiple isolation events are all recorded correctly."""
    controller = ZertzGameController(
        rings=37,
        headless=True,
        log_notation=True,
        seed=888
    )

    # We'll manually create a sequence of moves, some with isolation
    moves_with_isolation = []

    class Task:
        delay_time = 0

    task = Task()

    # Move 1: Normal move (no isolation)
    board = controller.session.game.board

    # Place a marble without isolation
    action1 = (0, board._2d_to_flat(*board.str_to_index("D4")),
              board._2d_to_flat(*board.str_to_index("A1")))

    _, action_dict1 = controller.session.game.action_to_str('PUT', action1)
    notation1 = controller.session.game.action_to_notation(action_dict1, None)
    controller.logger.buffer_notation(notation1)
    action_result1 = controller.session.game.take_action('PUT', action1)

    if action_result1.is_isolation():
        notation1 = controller.session.game.action_to_notation(action_dict1, action_result1.captured_marbles)
        controller.logger.update_buffered_notation(notation1)
        moves_with_isolation.append(notation1)
    else:
        moves_with_isolation.append(notation1)

    # Move 2: Set up and execute isolation
    # Create first isolation scenario
    g1_idx = board.str_to_index("G1")
    board.state[board.MARBLE_LAYERS.start, *g1_idx] = 1  # white
    board.state[board.RING_LAYER, *board.str_to_index("G2")] = 0
    board.state[board.RING_LAYER, *board.str_to_index("F1")] = 0

    action2 = (1, board._2d_to_flat(*board.str_to_index("E5")),
              board._2d_to_flat(*board.str_to_index("F2")))

    _, action_dict2 = controller.session.game.action_to_str('PUT', action2)
    notation2 = controller.session.game.action_to_notation(action_dict2, None)
    controller.logger.buffer_notation(notation2)  # Flushes notation1
    action_result2 = controller.session.game.take_action('PUT', action2)

    if action_result2.is_isolation() and action_result2.has_captures():
        notation2 = controller.session.game.action_to_notation(action_dict2, action_result2.captured_marbles)
        controller.logger.update_buffered_notation(notation2)
        moves_with_isolation.append(notation2)
    else:
        moves_with_isolation.append(notation2)

    # Close to flush final notation
    controller._close_log_file()

    # Verify file
    notation_file = f"zertzlog_888_notation.txt"
    moves = read_notation_file(notation_file)

    # Should have 2 moves
    assert len(moves) == 2
    # Second move should have isolation
    assert ' x ' in moves[1]


def test_isolation_at_game_end(temp_dir):
    """Test that isolation notation is correct even at game end."""
    controller = ZertzGameController(
        rings=37,
        headless=True,
        log_notation=True,
        seed=777
    )

    board = controller.session.game.board

    # Set up a near-win condition
    # Give player 1 almost enough captures
    board.global_state[board.P1_CAP_W] = 2  # Needs 3
    board.global_state[board.P1_CAP_G] = 2
    board.global_state[board.P1_CAP_B] = 2

    # Create isolation that will win the game
    g1_idx = board.str_to_index("G1")
    board.state[board.MARBLE_LAYERS.start, *g1_idx] = 1  # white
    board.state[board.RING_LAYER, *board.str_to_index("G2")] = 0
    board.state[board.RING_LAYER, *board.str_to_index("F1")] = 0

    action = (0, board._2d_to_flat(*board.str_to_index("D4")),
             board._2d_to_flat(*board.str_to_index("F2")))

    _, action_dict = controller.session.game.action_to_str('PUT', action)
    notation = controller.session.game.action_to_notation(action_dict, None)
    controller.logger.buffer_notation(notation)
    action_result = controller.session.game.take_action('PUT', action)

    if action_result.is_isolation() and action_result.has_captures():
        notation = controller.session.game.action_to_notation(action_dict, action_result.captured_marbles)
        controller.logger.update_buffered_notation(notation)

    # Close log
    controller._close_log_file()

    # Verify notation file
    notation_file = f"zertzlog_777_notation.txt"
    moves = read_notation_file(notation_file)

    # Should have the move with isolation
    assert len(moves) > 0
    assert ' x ' in moves[-1]


# ============================================================================
# Edge Case Integration Tests
# ============================================================================

def test_notation_without_logging_enabled(temp_dir):
    """Test that controller works correctly when notation logging is disabled."""
    controller = ZertzGameController(
        rings=37,
        headless=True,
        log_notation=False,  # Disabled
        seed=666
    )

    # Set up isolation
    board = controller.session.game.board
    action = create_isolation_scenario(board)

    _, action_dict = controller.session.game.action_to_str('PUT', action)
    notation = controller.session.game.action_to_notation(action_dict, None)

    # Should not crash even though logging is disabled
    controller.logger.buffer_notation(notation)

    action_result = controller.session.game.take_action('PUT', action)

    if action_result.is_isolation():
        notation = controller.session.game.action_to_notation(action_dict, action_result.captured_marbles)
        controller.logger.update_buffered_notation(notation)

    controller._close_log_file()

    # No notation file should be created
    notation_file = f"zertzlog_666_notation.txt"
    assert not os.path.exists(notation_file)


def test_vacant_ring_isolation_not_in_notation(temp_dir):
    """Test that isolated vacant rings don't appear in notation."""
    controller = ZertzGameController(
        rings=37,
        headless=True,
        log_notation=True,
        seed=555
    )

    board = controller.session.game.board

    # Set up G1 to be isolated but VACANT (no marble)
    board.state[board.RING_LAYER, *board.str_to_index("G2")] = 0
    board.state[board.RING_LAYER, *board.str_to_index("F1")] = 0

    action = (0, board._2d_to_flat(*board.str_to_index("D4")),
             board._2d_to_flat(*board.str_to_index("F2")))

    _, action_dict = controller.session.game.action_to_str('PUT', action)
    notation = controller.session.game.action_to_notation(action_dict, None)
    controller.logger.buffer_notation(notation)
    action_result = controller.session.game.take_action('PUT', action)

    # Vacant rings return ActionResult with empty captured_marbles
    if action_result.is_isolation() and action_result.has_captures():
        notation = controller.session.game.action_to_notation(action_dict, action_result.captured_marbles)
        controller.logger.update_buffered_notation(notation)

    controller._close_log_file()

    # Check notation file
    notation_file = f"zertzlog_555_notation.txt"
    moves = read_notation_file(notation_file)

    # Move should NOT have isolation notation (vacant ring)
    assert all(' x ' not in move for move in moves)


def test_blitz_variant_notation_with_isolation(temp_dir):
    """Test that blitz variant correctly handles isolation notation."""
    controller = ZertzGameController(
        rings=37,
        headless=True,
        log_notation=True,
        seed=444
    )

    # Manually set blitz flag
    controller.session.blitz = True

    board = controller.session.game.board
    action = create_isolation_scenario(board)

    _, action_dict = controller.session.game.action_to_str('PUT', action)
    notation = controller.session.game.action_to_notation(action_dict, None)
    controller.logger.buffer_notation(notation)
    action_result = controller.session.game.take_action('PUT', action)

    if action_result.is_isolation() and action_result.has_captures():
        notation = controller.session.game.action_to_notation(action_dict, action_result.captured_marbles)
        controller.logger.update_buffered_notation(notation)

    # Close with blitz flag
    controller._close_log_file()

    # Verify file exists (blitz should still log)
    # Note: filename might be different for blitz
    files = os.listdir('.')
    notation_files = [f for f in files if f.endswith('_notation.txt')]

    assert len(notation_files) > 0