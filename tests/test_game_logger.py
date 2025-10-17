"""
Unit tests for GameLogger.

Tests the pluggable writer system for logging game actions in different formats.
"""

import pytest
import sys
import tempfile
import os
import numpy as np
from pathlib import Path
from io import StringIO
from unittest.mock import Mock

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from controller.game_logger import GameLogger
from game.writers import TranscriptWriter, NotationWriter
from game.zertz_game import ZertzGame


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def game():
    """Create a standard game for testing."""
    return ZertzGame(rings=37)


@pytest.fixture
def mock_session():
    """Create a mock session for testing."""
    session = Mock()
    session.is_replay_mode.return_value = False
    session.blitz = False
    session.get_seed.return_value = 12345
    return session


# ============================================================================
# GameLogger Basic Tests
# ============================================================================


def test_logger_with_no_writers(mock_session):
    """Test that logger works with no writers."""
    logger = GameLogger(session=mock_session)
    logger.start_log(seed=12345, rings=37)
    logger.log_action(player_num=1, action_dict={"action": "PUT", "marble": "w", "dst": "D4", "remove": "B2"})
    logger.end_log()
    # Should not crash


def test_logger_start_log_writes_headers(temp_dir, mock_session):
    """Test that start_log writes headers to all writers."""
    os.chdir(temp_dir)

    # Create writers
    transcript_file = open("test_transcript.txt", "w")
    notation_file = open("test_notation.txt", "w")

    writers = [
        TranscriptWriter(transcript_file),
        NotationWriter(notation_file)
    ]

    # Create logger with session and manually add writers
    logger = GameLogger(session=mock_session)
    for writer in writers:
        logger.add_writer(writer)

    logger.start_log(seed=12345, rings=37, blitz=False)

    # Close writers manually to ensure flush
    for writer in writers:
        writer.close()

    # Check transcript header
    with open("test_transcript.txt", "r") as f:
        content = f.read()
        assert "# Seed: 12345" in content
        assert "# Rings: 37" in content

    # Check notation header
    with open("test_notation.txt", "r") as f:
        content = f.read()
        assert content.strip() == "37"


def test_logger_start_log_blitz_variant(temp_dir, mock_session):
    """Test that blitz variant is correctly noted in files."""
    os.chdir(temp_dir)

    transcript_file = open("test_transcript.txt", "w")
    notation_file = open("test_notation.txt", "w")

    writers = [
        TranscriptWriter(transcript_file),
        NotationWriter(notation_file)
    ]

    # Create logger with session and manually add writers
    logger = GameLogger(session=mock_session)
    for writer in writers:
        logger.add_writer(writer)

    logger.start_log(seed=12345, rings=37, blitz=True)

    for writer in writers:
        writer.close()

    # Check transcript
    with open("test_transcript.txt", "r") as f:
        content = f.read()
        assert "# Variant: Blitz" in content

    # Check notation
    with open("test_notation.txt", "r") as f:
        content = f.read()
        assert "37 Blitz" in content


# ============================================================================
# Log Action Tests
# ============================================================================


def test_log_action_writes_to_transcript(temp_dir, mock_session):
    """Test that log_action writes action to transcript file."""
    os.chdir(temp_dir)

    transcript_file = open("test_transcript.txt", "w")
    writer = TranscriptWriter(transcript_file)

    logger = GameLogger(session=mock_session)
    logger.add_writer(writer)
    logger.start_log(seed=12345, rings=37)

    action_dict = {"action": "PUT", "marble": "w", "dst": "D4", "remove": "B2"}
    logger.log_action(player_num=1, action_dict=action_dict)

    logger.end_log()
    writer.close()

    with open("test_transcript.txt", "r") as f:
        content = f.read()
        assert "Player 1:" in content
        assert "'action': 'PUT'" in content


def test_log_action_writes_to_notation(temp_dir, mock_session):
    """Test that log_action writes notation to notation file."""
    os.chdir(temp_dir)

    notation_file = open("test_notation.txt", "w")
    writer = NotationWriter(notation_file)

    logger = GameLogger(session=mock_session)
    logger.add_writer(writer)
    logger.start_log(seed=12345, rings=37)

    # Simple PUT action
    action_dict = {"action": "PUT", "marble": "w", "dst": "D4", "remove": "B2"}
    logger.log_action(player_num=1, action_dict=action_dict, action_result=None)

    logger.end_log()
    writer.close()

    with open("test_notation.txt", "r") as f:
        lines = [l.strip() for l in f.readlines()]
        moves = [l for l in lines[1:] if l]  # Skip header line
        assert "Wd4,b2" in moves


def test_log_action_with_pass(temp_dir, mock_session):
    """Test that PASS actions are logged correctly."""
    os.chdir(temp_dir)

    notation_file = open("test_notation.txt", "w")
    writer = NotationWriter(notation_file)

    logger = GameLogger(session=mock_session)
    logger.add_writer(writer)
    logger.start_log(seed=12345, rings=37)

    action_dict = {"action": "PASS"}
    logger.log_action(player_num=1, action_dict=action_dict)

    logger.end_log()
    writer.close()

    with open("test_notation.txt", "r") as f:
        content = f.read()
        assert "-" in content


def test_log_action_multiple_writers(temp_dir, mock_session):
    """Test that actions are logged to multiple writers simultaneously."""
    os.chdir(temp_dir)

    transcript_file = open("test_transcript.txt", "w")
    notation_file = open("test_notation.txt", "w")

    writers = [
        TranscriptWriter(transcript_file),
        NotationWriter(notation_file)
    ]

    logger = GameLogger(session=mock_session)
    for writer in writers:
        logger.add_writer(writer)
    logger.start_log(seed=12345, rings=37)

    action_dict = {"action": "PUT", "marble": "w", "dst": "D4", "remove": "B2"}
    logger.log_action(player_num=1, action_dict=action_dict)

    logger.end_log()
    for writer in writers:
        writer.close()

    # Check both files
    with open("test_transcript.txt", "r") as f:
        assert "Player 1:" in f.read()

    with open("test_notation.txt", "r") as f:
        assert "Wd4,b2" in f.read()


# ============================================================================
# Screen Output Tests
# ============================================================================


def test_logger_with_string_io(mock_session):
    """Test that logger works with StringIO for screen output."""
    transcript_stream = StringIO()
    notation_stream = StringIO()

    writers = [
        TranscriptWriter(transcript_stream),
        NotationWriter(notation_stream)
    ]

    logger = GameLogger(session=mock_session)
    for writer in writers:
        logger.add_writer(writer)
    logger.start_log(seed=12345, rings=37)

    action_dict = {"action": "PUT", "marble": "w", "dst": "D4", "remove": "B2"}
    logger.log_action(player_num=1, action_dict=action_dict)

    # Get output BEFORE calling end_log (which closes the streams)
    transcript_output = transcript_stream.getvalue()
    notation_output = notation_stream.getvalue()

    logger.end_log()

    # Check transcript output
    assert "# Seed: 12345" in transcript_output
    assert "Player 1:" in transcript_output

    # Check notation output
    assert "37" in notation_output
    assert "Wd4,b2" in notation_output


# ============================================================================
# End Game Tests
# ============================================================================


def test_end_log_writes_footer(temp_dir, mock_session):
    """Test that end_log writes footer to transcript file."""
    os.chdir(temp_dir)

    transcript_file = open("test_transcript.txt", "w")
    writer = TranscriptWriter(transcript_file)

    logger = GameLogger(session=mock_session)
    logger.add_writer(writer)
    logger.start_log(seed=12345, rings=37)

    # Create a mock game with state
    mock_game = Mock()
    mock_game.board.state = np.zeros((20, 7, 7))

    logger.end_log(game=mock_game)
    writer.close()

    with open("test_transcript.txt", "r") as f:
        content = f.read()
        assert "# Final game state:" in content
        assert "# Board state:" in content


# ============================================================================
# Integration Tests
# ============================================================================


def test_notation_sequence(temp_dir, mock_session):
    """Test logging a sequence of actions in notation format."""
    os.chdir(temp_dir)

    notation_file = open("test_notation.txt", "w")
    writer = NotationWriter(notation_file)

    logger = GameLogger(session=mock_session)
    logger.add_writer(writer)
    logger.start_log(seed=12345, rings=37)

    # Log multiple actions
    logger.log_action(1, {"action": "PUT", "marble": "w", "dst": "D4", "remove": "B2"})
    logger.log_action(2, {"action": "PUT", "marble": "g", "dst": "E5", "remove": "C3"})
    logger.log_action(1, {"action": "PUT", "marble": "b", "dst": "F6", "remove": "D1"})

    logger.end_log()
    writer.close()

    # Verify all notations written in order
    with open("test_notation.txt", "r") as f:
        lines = [l.strip() for l in f.readlines()]
        moves = [l for l in lines[1:] if l]  # Skip header
        assert "Wd4,b2" in moves[0]
        assert "Ge5,c3" in moves[1]
        assert "Bf6,d1" in moves[2]


def test_add_writer_dynamically(mock_session):
    """Test adding writers dynamically after logger creation."""
    logger = GameLogger(session=mock_session)

    stream = StringIO()
    writer = TranscriptWriter(stream)
    logger.add_writer(writer)

    logger.start_log(seed=12345, rings=37)
    logger.log_action(1, {"action": "PUT", "marble": "w", "dst": "D4", "remove": ""})

    # Get output BEFORE calling end_log (which closes the stream)
    output = stream.getvalue()

    logger.end_log()

    assert "# Seed: 12345" in output
    assert "Player 1:" in output


def test_remove_writer(mock_session):
    """Test removing writers from logger."""
    stream = StringIO()
    writer = TranscriptWriter(stream)

    logger = GameLogger(session=mock_session)
    logger.add_writer(writer)
    logger.remove_writer(writer)

    logger.start_log(seed=12345, rings=37)
    logger.log_action(1, {"action": "PUT", "marble": "w", "dst": "D4", "remove": ""})
    logger.end_log()

    # Stream should be empty since writer was removed
    output = stream.getvalue()
    assert output == ""