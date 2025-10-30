"""Test that player names are displayed correctly in text output."""

import tempfile
import sys
from pathlib import Path
from io import StringIO

sys.path.insert(0, str(Path(__file__).parent.parent))

from game.writers import TranscriptWriter
from game.utils.player_utils import format_player_name


def test_format_player_name_without_custom_name():
    """Test formatting player display name without custom name."""
    assert format_player_name(1) == "Player 1"
    assert format_player_name(2) == "Player 2"


def test_format_player_name_with_custom_name():
    """Test formatting player display name with custom name."""
    assert format_player_name(1, "Alice") == "Player 1 (Alice)"
    assert format_player_name(2, "Bob") == "Player 2 (Bob)"


def test_format_player_name_with_none():
    """Test formatting player display name with None."""
    assert format_player_name(1, None) == "Player 1"
    assert format_player_name(2, None) == "Player 2"


def test_transcript_writer_displays_player_names():
    """Test that transcript writer includes player names in action lines."""
    stream = StringIO()
    writer = TranscriptWriter(stream)

    # Write header with player names
    writer.write_header(seed=12345, rings=37, blitz=False, player1_name="Alice", player2_name="Bob")

    # Write actions
    writer.write_action(1, {'action': 'PUT', 'marble': 'w', 'dst': 'D4', 'remove': ''})
    writer.write_action(2, {'action': 'PUT', 'marble': 'g', 'dst': 'E3', 'remove': 'A1'})

    output = stream.getvalue()

    # Check that player names appear in action lines
    assert "Player 1 (Alice):" in output
    assert "Player 2 (Bob):" in output


def test_transcript_writer_without_player_names():
    """Test that transcript writer works without player names (backward compatibility)."""
    stream = StringIO()
    writer = TranscriptWriter(stream)

    # Write header without player names
    writer.write_header(seed=12345, rings=37, blitz=False)

    # Write actions
    writer.write_action(1, {'action': 'PUT', 'marble': 'w', 'dst': 'D4', 'remove': ''})
    writer.write_action(2, {'action': 'PUT', 'marble': 'g', 'dst': 'E3', 'remove': 'A1'})

    output = stream.getvalue()

    # Check that only generic player numbers appear (no parentheses)
    assert "Player 1:" in output
    assert "Player 2:" in output
    assert "Player 1 (" not in output
    assert "Player 2 (" not in output


def test_transcript_writer_with_only_player1_name():
    """Test that transcript writer handles only player 1 having a name."""
    stream = StringIO()
    writer = TranscriptWriter(stream)

    # Write header with only player1 name
    writer.write_header(seed=12345, rings=37, blitz=False, player1_name="Alice", player2_name=None)

    # Write actions
    writer.write_action(1, {'action': 'PUT', 'marble': 'w', 'dst': 'D4', 'remove': ''})
    writer.write_action(2, {'action': 'PUT', 'marble': 'g', 'dst': 'E3', 'remove': 'A1'})

    output = stream.getvalue()

    # Check that player 1 has name but player 2 doesn't
    assert "Player 1 (Alice):" in output
    assert "Player 2:" in output
    assert "Player 2 (" not in output