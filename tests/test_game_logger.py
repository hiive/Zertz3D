"""
Unit tests for GameLogger buffering system (Temporal Coupling Issue 2).

Tests the notation buffering system that enables deferred notation updates
when isolation occurs after action execution.
"""

import pytest
import sys
import tempfile
import os
import numpy as np
from pathlib import Path
from unittest.mock import Mock, MagicMock, mock_open, patch

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from controller.game_logger import GameLogger
from game.zertz_game import ZertzGame


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def logger():
    """Create a GameLogger with both logging types enabled."""
    return GameLogger(log_to_file=True, log_notation=True)


@pytest.fixture
def logger_no_files():
    """Create a GameLogger with file logging disabled."""
    return GameLogger(log_to_file=False, log_notation=False)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def game():
    """Create a standard game for testing."""
    return ZertzGame(rings=37)


# ============================================================================
# Buffer Notation Tests
# ============================================================================

def test_buffer_notation_stores_notation(logger_no_files):
    """Test that buffer_notation stores notation in pending_notation."""
    notation = "Wd4"
    logger_no_files.buffer_notation(notation)

    assert logger_no_files.pending_notation == notation


def test_buffer_notation_flushes_previous(logger_no_files):
    """Test that buffering new notation flushes previous notation."""
    # Buffer first notation
    logger_no_files.buffer_notation("Wd4")
    assert logger_no_files.pending_notation == "Wd4"

    # Buffer second notation - should replace first
    logger_no_files.buffer_notation("Ge5")
    assert logger_no_files.pending_notation == "Ge5"


def test_buffer_notation_writes_previous_to_file(logger, temp_dir):
    """Test that buffering new notation writes previous to file."""
    # Open files
    os.chdir(temp_dir)
    logger.open_log_files(seed=12345, rings=37)

    # Buffer first notation
    logger.buffer_notation("Wd4")

    # Buffer second notation - should write first to file
    logger.buffer_notation("Ge5")

    # Close files with proper mock using numpy array
    mock_game = Mock()
    mock_game.board.state = np.zeros((20, 7, 7))  # Simplified board state
    logger.close_log_files(mock_game)

    # Verify first notation was written, second is in file from close
    with open(logger.notation_filename, 'r') as f:
        lines = f.readlines()
        # Skip first line (board size)
        assert "Wd4\n" in lines
        assert "Ge5\n" in lines


def test_buffer_notation_none_pending_does_not_write(logger_no_files):
    """Test that buffering with no pending notation doesn't crash."""
    # No pending notation yet
    assert logger_no_files.pending_notation is None

    # Should not crash
    logger_no_files.buffer_notation("Wd4")
    assert logger_no_files.pending_notation == "Wd4"


# ============================================================================
# Update Buffered Notation Tests
# ============================================================================

def test_update_buffered_notation_replaces_pending(logger_no_files):
    """Test that update_buffered_notation replaces pending notation."""
    # Buffer initial notation
    logger_no_files.buffer_notation("Wd4")

    # Update with isolation
    logger_no_files.update_buffered_notation("Wd4 x Bg1")

    assert logger_no_files.pending_notation == "Wd4 x Bg1"


def test_update_buffered_notation_without_buffer(logger_no_files):
    """Test updating notation when nothing is buffered."""
    # No buffered notation
    assert logger_no_files.pending_notation is None

    # Update should still work
    logger_no_files.update_buffered_notation("Wd4 x Bg1")
    assert logger_no_files.pending_notation == "Wd4 x Bg1"


def test_update_keeps_notation_pending(logger, temp_dir):
    """Test that update doesn't write to file, just updates buffer."""
    os.chdir(temp_dir)
    logger.open_log_files(seed=12345, rings=37)

    # Buffer notation
    logger.buffer_notation("Wd4")

    # Update notation
    logger.update_buffered_notation("Wd4 x Bg1")

    # Close and check file with proper mock using numpy array
    mock_game = Mock()
    mock_game.board.state = np.zeros((20, 7, 7))  # Simplified board state
    logger.close_log_files(mock_game)

    with open(logger.notation_filename, 'r') as f:
        content = f.read()
        # Should have updated notation, not original
        assert "Wd4 x Bg1" in content
        # Should not have both versions
        lines = [l.strip() for l in content.split('\n') if l.strip() and not l.startswith('#')]
        assert lines.count("Wd4") == 0  # Original should not appear alone


# ============================================================================
# Close Log Files Tests
# ============================================================================

def test_close_writes_pending_notation(logger, temp_dir):
    """Test that closing files writes pending notation."""
    os.chdir(temp_dir)
    logger.open_log_files(seed=12345, rings=37)

    # Buffer notation
    logger.buffer_notation("Wd4")

    # Close files
    mock_game = Mock()
    mock_game.board.state = Mock()
    mock_game.board.state.__getitem__ = Mock(return_value=[[0]])

    logger.close_log_files(mock_game)

    # Check notation was written
    with open(logger.notation_filename, 'r') as f:
        content = f.read()
        assert "Wd4" in content


def test_close_clears_pending_notation(logger, temp_dir):
    """Test that closing files clears pending_notation."""
    os.chdir(temp_dir)
    logger.open_log_files(seed=12345, rings=37)

    logger.buffer_notation("Wd4")
    assert logger.pending_notation == "Wd4"

    # Close files
    mock_game = Mock()
    mock_game.board.state = Mock()
    mock_game.board.state.__getitem__ = Mock(return_value=[[0]])

    logger.close_log_files(mock_game)

    # Should be cleared
    assert logger.pending_notation is None


def test_close_with_no_pending_notation(logger, temp_dir):
    """Test closing files when no notation is pending."""
    os.chdir(temp_dir)
    logger.open_log_files(seed=12345, rings=37)

    # No pending notation
    assert logger.pending_notation is None

    mock_game = Mock()
    mock_game.board.state = Mock()
    mock_game.board.state.__getitem__ = Mock(return_value=[[0]])

    # Should not crash
    logger.close_log_files(mock_game)


# ============================================================================
# Integration Tests
# ============================================================================

def test_notation_sequence_without_isolation(logger, temp_dir):
    """Test typical notation sequence without isolation."""
    os.chdir(temp_dir)
    logger.open_log_files(seed=12345, rings=37)

    # Sequence of moves
    logger.buffer_notation("Wd4")
    logger.buffer_notation("Ge5")  # Flushes Wd4
    logger.buffer_notation("Bf6")  # Flushes Ge5

    # Close
    mock_game = Mock()
    mock_game.board.state = Mock()
    mock_game.board.state.__getitem__ = Mock(return_value=[[0]])
    logger.close_log_files(mock_game)  # Flushes Bf6

    # Verify all notations written in order
    with open(logger.notation_filename, 'r') as f:
        lines = [l.strip() for l in f.readlines()]
        # Skip first line (board size) and filter out empty lines and comments
        moves = [l for l in lines[1:] if l and not l.startswith('#')]
        assert moves == ["Wd4", "Ge5", "Bf6"]


def test_notation_sequence_with_isolation_update(logger, temp_dir):
    """Test notation sequence with isolation causing update."""
    os.chdir(temp_dir)
    logger.open_log_files(seed=12345, rings=37)

    # First move - normal
    logger.buffer_notation("Wd4")

    # Second move - initially without isolation
    logger.buffer_notation("Ge5")  # Flushes Wd4

    # Isolation detected after action - update buffered notation
    logger.update_buffered_notation("Ge5 x Wa1Wb2")

    # Third move
    logger.buffer_notation("Bf6")  # Flushes updated Ge5

    # Close
    mock_game = Mock()
    mock_game.board.state = Mock()
    mock_game.board.state.__getitem__ = Mock(return_value=[[0]])
    logger.close_log_files(mock_game)

    # Verify correct notations
    with open(logger.notation_filename, 'r') as f:
        lines = [l.strip() for l in f.readlines()]
        # Skip first line (board size) and filter out empty lines and comments
        moves = [l for l in lines[1:] if l and not l.startswith('#')]
        assert moves == ["Wd4", "Ge5 x Wa1Wb2", "Bf6"]


def test_disabled_notation_logging(logger_no_files):
    """Test that operations work when notation logging is disabled."""
    # All operations should work without errors
    logger_no_files.buffer_notation("Wd4")
    logger_no_files.update_buffered_notation("Wd4 x Bg1")
    logger_no_files.buffer_notation("Ge5")

    # Notation should be tracked even if not written
    assert logger_no_files.pending_notation == "Ge5"


def test_multiple_updates_to_same_notation(logger_no_files):
    """Test multiple updates to the same buffered notation."""
    logger_no_files.buffer_notation("Wd4")

    # Multiple updates (edge case, but should work)
    logger_no_files.update_buffered_notation("Wd4 x Wa1")
    logger_no_files.update_buffered_notation("Wd4 x Wa1Wb2")
    logger_no_files.update_buffered_notation("Wd4 x Wa1Wb2Gc3")

    # Should have final version
    assert logger_no_files.pending_notation == "Wd4 x Wa1Wb2Gc3"