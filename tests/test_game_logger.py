"""
Unit tests for GameLogger.

Tests the logging system where notation is generated once
with complete information (including isolation) and logged directly.
"""

import pytest
import sys
import tempfile
import os
import numpy as np
from pathlib import Path
from unittest.mock import Mock

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
# Open/Close Log Files Tests
# ============================================================================


def test_open_log_files_creates_files(logger, temp_dir):
    """Test that open_log_files creates both log files."""
    os.chdir(temp_dir)
    logger.open_log_files(seed=12345, rings=37)

    assert logger.log_file is not None
    assert logger.notation_file is not None
    assert os.path.exists(logger.log_filename)
    assert os.path.exists(logger.notation_filename)

    # Clean up
    mock_game = Mock()
    mock_game.board.state = np.zeros((20, 7, 7))
    logger.close_log_files(mock_game)


def test_open_log_files_writes_headers(logger, temp_dir):
    """Test that log files have correct headers."""
    os.chdir(temp_dir)
    logger.open_log_files(seed=12345, rings=37)

    # Flush files to ensure content is written
    logger.log_file.flush()
    logger.notation_file.flush()

    # Check action log header
    with open(logger.log_filename, "r") as f:
        content = f.read()
        assert "# Seed: 12345" in content
        assert "# Rings: 37" in content

    # Check notation log header
    with open(logger.notation_filename, "r") as f:
        content = f.read()
        assert content.strip() == "37"

    # Clean up
    mock_game = Mock()
    mock_game.board.state = np.zeros((20, 7, 7))
    logger.close_log_files(mock_game)


def test_open_log_files_blitz_variant(logger, temp_dir):
    """Test that blitz variant is correctly noted in files."""
    os.chdir(temp_dir)
    logger.open_log_files(seed=12345, rings=37, blitz=True)

    # Flush files to ensure content is written
    logger.log_file.flush()
    logger.notation_file.flush()

    # Check action log
    with open(logger.log_filename, "r") as f:
        content = f.read()
        assert "# Variant: Blitz" in content

    # Check notation log
    with open(logger.notation_filename, "r") as f:
        content = f.read()
        assert "37 Blitz" in content

    # Clean up
    mock_game = Mock()
    mock_game.board.state = np.zeros((20, 7, 7))
    logger.close_log_files(mock_game)


def test_close_log_files_appends_game_state(logger, temp_dir):
    """Test that closing log files appends final game state."""
    os.chdir(temp_dir)
    logger.open_log_files(seed=12345, rings=37)

    mock_game = Mock()
    mock_game.board.state = np.zeros((20, 7, 7))
    logger.close_log_files(mock_game)

    with open(logger.log_filename, "r") as f:
        content = f.read()
        assert "# Final game state:" in content
        assert "# Board state:" in content


def test_close_log_files_clears_file_handles(logger, temp_dir):
    """Test that closing files clears file handles."""
    os.chdir(temp_dir)
    logger.open_log_files(seed=12345, rings=37)

    mock_game = Mock()
    mock_game.board.state = np.zeros((20, 7, 7))
    logger.close_log_files(mock_game)

    assert logger.log_file is None
    assert logger.notation_file is None


# ============================================================================
# Log Action Tests
# ============================================================================


def test_log_action_writes_to_file(logger, temp_dir):
    """Test that log_action writes action to file."""
    os.chdir(temp_dir)
    logger.open_log_files(seed=12345, rings=37)

    action_dict = {"action": "PUT", "marble": "w", "dst": "D4", "remove": "B2"}
    logger.log_action(player_num=1, action_dict=action_dict)

    mock_game = Mock()
    mock_game.board.state = np.zeros((20, 7, 7))
    logger.close_log_files(mock_game)

    with open(logger.log_filename, "r") as f:
        content = f.read()
        assert "Player 1:" in content
        assert "'action': 'PUT'" in content


def test_log_action_disabled_when_no_file(logger_no_files):
    """Test that log_action doesn't crash when logging is disabled."""
    action_dict = {"action": "PUT", "marble": "w", "dst": "D4", "remove": "B2"}
    # Should not crash
    logger_no_files.log_action(player_num=1, action_dict=action_dict)


# ============================================================================
# Log Notation Tests
# ============================================================================


def test_log_notation_writes_to_file(logger, temp_dir):
    """Test that log_notation writes notation to file."""
    os.chdir(temp_dir)
    logger.open_log_files(seed=12345, rings=37)

    logger.log_notation("Wd4,b2")
    logger.log_notation("Ge5,c3")

    mock_game = Mock()
    mock_game.board.state = np.zeros((20, 7, 7))
    logger.close_log_files(mock_game)

    with open(logger.notation_filename, "r") as f:
        lines = [l.strip() for l in f.readlines()]
        moves = [l for l in lines[1:] if l]  # Skip header line
        assert "Wd4,b2" in moves
        assert "Ge5,c3" in moves


def test_log_notation_with_isolation(logger, temp_dir):
    """Test that notation with isolation is logged correctly."""
    os.chdir(temp_dir)
    logger.open_log_files(seed=12345, rings=37)

    # Notation with isolation marker
    logger.log_notation("Wd4,b2 x Wa1Wb2")

    mock_game = Mock()
    mock_game.board.state = np.zeros((20, 7, 7))
    logger.close_log_files(mock_game)

    with open(logger.notation_filename, "r") as f:
        content = f.read()
        assert "Wd4,b2 x Wa1Wb2" in content


def test_log_notation_disabled_when_no_file(logger_no_files):
    """Test that log_notation doesn't crash when logging is disabled."""
    # Should not crash
    logger_no_files.log_notation("Wd4,b2")


# ============================================================================
# Integration Tests
# ============================================================================


def test_notation_sequence_simple(logger, temp_dir):
    """Test logging a sequence of notations."""
    os.chdir(temp_dir)
    logger.open_log_files(seed=12345, rings=37)

    # Log multiple notations
    logger.log_notation("Wd4,b2")
    logger.log_notation("Ge5,c3")
    logger.log_notation("Bf6,d1")

    mock_game = Mock()
    mock_game.board.state = np.zeros((20, 7, 7))
    logger.close_log_files(mock_game)

    # Verify all notations written in order
    with open(logger.notation_filename, "r") as f:
        lines = [l.strip() for l in f.readlines()]
        moves = [l for l in lines[1:] if l]  # Skip header
        assert moves == ["Wd4,b2", "Ge5,c3", "Bf6,d1"]


def test_notation_with_mixed_actions(logger, temp_dir):
    """Test logging notations including captures and isolation."""
    os.chdir(temp_dir)
    logger.open_log_files(seed=12345, rings=37)

    # Mixed action types
    logger.log_notation("Wd4,b2")  # PUT
    logger.log_notation("x c4Wa2")  # CAP
    logger.log_notation("Ge5,c3 x Wa1")  # PUT with isolation
    logger.log_notation("-")  # PASS

    mock_game = Mock()
    mock_game.board.state = np.zeros((20, 7, 7))
    logger.close_log_files(mock_game)

    with open(logger.notation_filename, "r") as f:
        lines = [l.strip() for l in f.readlines()]
        moves = [l for l in lines[1:] if l]
        assert moves == ["Wd4,b2", "x c4Wa2", "Ge5,c3 x Wa1", "-"]


def test_multiple_games_same_logger(logger, temp_dir):
    """Test that logger can handle multiple game sessions."""
    os.chdir(temp_dir)

    # First game
    logger.open_log_files(seed=12345, rings=37)
    logger.log_notation("Wd4,b2")
    mock_game = Mock()
    mock_game.board.state = np.zeros((20, 7, 7))
    logger.close_log_files(mock_game)

    # Second game
    logger.open_log_files(seed=67890, rings=37)
    logger.log_notation("Ge5,c3")
    logger.close_log_files(mock_game)

    # Both files should exist
    assert os.path.exists("zertzlog_12345_notation.txt")
    assert os.path.exists("zertzlog_67890_notation.txt")
