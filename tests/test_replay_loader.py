"""Tests for TranscriptLoader functionality."""

import tempfile
import os
import pytest

from game.loaders import TranscriptLoader
from game.zertz_board import ZertzBoard
from game.zertz_game import (
    STANDARD_MARBLES,
    BLITZ_MARBLES,
    STANDARD_WIN_CONDITIONS,
    BLITZ_WIN_CONDITIONS,
)


class TestTranscriptLoader:
    """Test suite for TranscriptLoader class."""

    def test_basic_37_ring_replay(self):
        """Test loading a basic 37-ring game replay."""
        replay_content = """# Test replay file
Player 1: {'action': 'PUT', 'marble': 'w', 'dst': 'D4', 'remove': ''}
Player 2: {'action': 'PUT', 'marble': 'g', 'dst': 'E3', 'remove': 'A1'}
Player 1: {'action': 'CAP', 'marble': 'g', 'src': 'E3', 'capture': 'w', 'dst': 'C5', 'cap': 'D4'}
"""

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write(replay_content)
            temp_file = f.name

        try:
            loader = TranscriptLoader(temp_file)
            p1_actions, p2_actions = loader.load()

            # Verify board size detection
            assert loader.detected_rings == ZertzBoard.SMALL_BOARD_37

            # Verify standard variant
            assert not loader.blitz
            assert loader.marbles == STANDARD_MARBLES
            assert loader.win_condition == STANDARD_WIN_CONDITIONS

            # Verify action counts
            assert len(p1_actions) == 2
            assert len(p2_actions) == 1

            # Verify action content
            assert p1_actions[0]["action"] == "PUT"
            assert p1_actions[0]["marble"] == "w"
            assert p1_actions[0]["dst"] == "D4"

            assert p2_actions[0]["action"] == "PUT"
            assert p2_actions[0]["marble"] == "g"

            assert p1_actions[1]["action"] == "CAP"
            assert p1_actions[1]["src"] == "E3"
        finally:
            os.unlink(temp_file)

    def test_48_ring_board_detection(self):
        """Test detection of 48-ring board (columns A-H)."""
        replay_content = """Player 1: {'action': 'PUT', 'marble': 'w', 'dst': 'H1', 'remove': ''}
Player 2: {'action': 'PUT', 'marble': 'g', 'dst': 'H8', 'remove': ''}
"""

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write(replay_content)
            temp_file = f.name

        try:
            loader = TranscriptLoader(temp_file)
            p1_actions, p2_actions = loader.load()

            assert loader.detected_rings == ZertzBoard.MEDIUM_BOARD_48
        finally:
            os.unlink(temp_file)

    def test_61_ring_board_detection(self):
        """Test detection of 61-ring board (columns A-J, skipping I)."""
        replay_content = """Player 1: {'action': 'PUT', 'marble': 'w', 'dst': 'J1', 'remove': ''}
Player 2: {'action': 'PUT', 'marble': 'g', 'dst': 'J9', 'remove': ''}
"""

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write(replay_content)
            temp_file = f.name

        try:
            loader = TranscriptLoader(temp_file)
            p1_actions, p2_actions = loader.load()

            assert loader.detected_rings == ZertzBoard.LARGE_BOARD_61
        finally:
            os.unlink(temp_file)

    def test_blitz_variant_detection_from_file(self):
        """Test blitz variant detection from file header."""
        replay_content = """# Variant: Blitz
Player 1: {'action': 'PUT', 'marble': 'w', 'dst': 'D4', 'remove': ''}
Player 2: {'action': 'PUT', 'marble': 'g', 'dst': 'E3', 'remove': ''}
"""

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write(replay_content)
            temp_file = f.name

        try:
            loader = TranscriptLoader(temp_file)
            loader.load()

            assert loader.blitz
            assert loader.marbles == BLITZ_MARBLES
            assert loader.win_condition == BLITZ_WIN_CONDITIONS
        finally:
            os.unlink(temp_file)

    def test_pass_action_parsing(self):
        """Test parsing of PASS actions."""
        replay_content = """Player 1: {'action': 'PUT', 'marble': 'w', 'dst': 'D4', 'remove': ''}
Player 2: {'action': 'PASS'}
Player 1: {'action': 'PUT', 'marble': 'g', 'dst': 'E3', 'remove': ''}
"""

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write(replay_content)
            temp_file = f.name

        try:
            loader = TranscriptLoader(temp_file)
            p1_actions, p2_actions = loader.load()

            assert len(p1_actions) == 2
            assert len(p2_actions) == 1
            assert p2_actions[0]["action"] == "PASS"
        finally:
            os.unlink(temp_file)

    def test_comment_lines_ignored(self):
        """Test that comment lines are properly ignored."""
        replay_content = """# This is a comment
# Another comment
Player 1: {'action': 'PUT', 'marble': 'w', 'dst': 'D4', 'remove': ''}
# Mid-game comment
Player 2: {'action': 'PUT', 'marble': 'g', 'dst': 'E3', 'remove': ''}
"""

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write(replay_content)
            temp_file = f.name

        try:
            loader = TranscriptLoader(temp_file)
            p1_actions, p2_actions = loader.load()

            assert len(p1_actions) == 1
            assert len(p2_actions) == 1
        finally:
            os.unlink(temp_file)

    def test_empty_lines_ignored(self):
        """Test that empty lines are properly ignored."""
        replay_content = """
Player 1: {'action': 'PUT', 'marble': 'w', 'dst': 'D4', 'remove': ''}

Player 2: {'action': 'PUT', 'marble': 'g', 'dst': 'E3', 'remove': ''}

"""

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write(replay_content)
            temp_file = f.name

        try:
            loader = TranscriptLoader(temp_file)
            p1_actions, p2_actions = loader.load()

            assert len(p1_actions) == 1
            assert len(p2_actions) == 1
        finally:
            os.unlink(temp_file)

    def test_actions_with_all_fields(self):
        """Test parsing actions with all possible fields."""
        replay_content = """Player 1: {'action': 'PUT', 'marble': 'b', 'dst': 'D4', 'remove': 'G1'}
Player 2: {'action': 'CAP', 'marble': 'b', 'src': 'D4', 'capture': 'w', 'dst': 'F6', 'cap': 'E5'}
"""

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write(replay_content)
            temp_file = f.name

        try:
            loader = TranscriptLoader(temp_file)
            p1_actions, p2_actions = loader.load()

            # Verify PUT action
            put_action = p1_actions[0]
            assert put_action["action"] == "PUT"
            assert put_action["marble"] == "b"
            assert put_action["dst"] == "D4"
            assert put_action["remove"] == "G1"

            # Verify CAP action
            cap_action = p2_actions[0]
            assert cap_action["action"] == "CAP"
            assert cap_action["marble"] == "b"
            assert cap_action["src"] == "D4"
            assert cap_action["capture"] == "w"
            assert cap_action["dst"] == "F6"
            assert cap_action["cap"] == "E5"
        finally:
            os.unlink(temp_file)

    def test_board_size_detection_uses_all_coordinate_fields(self):
        """Test that board size detection checks all coordinate fields (dst, src, remove, cap)."""
        replay_content = """Player 1: {'action': 'PUT', 'marble': 'w', 'dst': 'A1', 'remove': 'J9'}
"""

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write(replay_content)
            temp_file = f.name

        try:
            loader = TranscriptLoader(temp_file)
            p1_actions, p2_actions = loader.load()

            # Should detect 61-ring board from 'J' in remove field
            assert loader.detected_rings == ZertzBoard.LARGE_BOARD_61
        finally:
            os.unlink(temp_file)

    def test_lowercase_coordinates(self):
        """Test that lowercase coordinates are properly handled."""
        replay_content = """Player 1: {'action': 'PUT', 'marble': 'w', 'dst': 'd4', 'remove': 'g1'}
"""

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write(replay_content)
            temp_file = f.name

        try:
            loader = TranscriptLoader(temp_file)
            p1_actions, p2_actions = loader.load()

            # Should still work with lowercase
            assert loader.detected_rings == ZertzBoard.SMALL_BOARD_37
            assert p1_actions[0]["dst"] == "d4"
        finally:
            os.unlink(temp_file)

    def test_status_reporter_callback(self):
        """Test that status reporter callback is called."""
        replay_content = """Player 1: {'action': 'PUT', 'marble': 'w', 'dst': 'D4', 'remove': ''}
"""

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write(replay_content)
            temp_file = f.name

        try:
            messages = []

            def reporter(msg):
                messages.append(msg)

            loader = TranscriptLoader(temp_file, status_reporter=reporter)
            p1_actions, p2_actions = loader.load()

            # Should have received status messages
            assert len(messages) > 0
            assert any("Loading transcript" in msg for msg in messages)
            assert any("Detected board size" in msg for msg in messages)
        finally:
            os.unlink(temp_file)

    def test_set_status_reporter_after_init(self):
        """Test setting status reporter after initialization."""
        replay_content = """Player 1: {'action': 'PUT', 'marble': 'w', 'dst': 'D4', 'remove': ''}
"""

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write(replay_content)
            temp_file = f.name

        try:
            messages = []

            def reporter(msg):
                messages.append(msg)

            loader = TranscriptLoader(temp_file)
            loader.set_status_reporter(reporter)
            p1_actions, p2_actions = loader.load()

            # Should have received messages
            assert len(messages) > 0
        finally:
            os.unlink(temp_file)

    def test_empty_file(self):
        """Test handling of empty replay file."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("")
            temp_file = f.name

        try:
            loader = TranscriptLoader(temp_file)
            p1_actions, p2_actions = loader.load()

            # Empty file should return empty action lists
            assert len(p1_actions) == 0
            assert len(p2_actions) == 0

            # Should default to 37-ring board
            assert loader.detected_rings == ZertzBoard.SMALL_BOARD_37
        finally:
            os.unlink(temp_file)

    def test_only_comments_file(self):
        """Test file with only comments."""
        replay_content = """# Comment 1
# Comment 2
# Comment 3
"""

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write(replay_content)
            temp_file = f.name

        try:
            loader = TranscriptLoader(temp_file)
            p1_actions, p2_actions = loader.load()

            assert len(p1_actions) == 0
            assert len(p2_actions) == 0
            assert loader.detected_rings == ZertzBoard.SMALL_BOARD_37
        finally:
            os.unlink(temp_file)

    def test_variant_case_insensitive(self):
        """Test that variant detection is case insensitive."""
        replay_content = """# Variant: BLITZ
Player 1: {'action': 'PUT', 'marble': 'w', 'dst': 'D4', 'remove': ''}
"""

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write(replay_content)
            temp_file = f.name

        try:
            loader = TranscriptLoader(temp_file)
            p1_actions, p2_actions = loader.load()

            assert loader.blitz
        finally:
            os.unlink(temp_file)

    def test_file_not_found(self):
        """Test handling of non-existent file."""
        with pytest.raises(FileNotFoundError):
            loader = TranscriptLoader("/nonexistent/path/to/file.txt")
            loader.load()

    def test_malformed_action_dict(self):
        """Test handling of malformed action dictionary."""
        replay_content = """Player 1: {this is not valid python}
"""

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write(replay_content)
            temp_file = f.name

        try:
            loader = TranscriptLoader(temp_file)
            # Should raise SyntaxError or ValueError from ast.literal_eval
            with pytest.raises((SyntaxError, ValueError)):
                loader.load()
        finally:
            os.unlink(temp_file)

    def test_invalid_player_number(self):
        """Test handling of invalid player numbers."""
        replay_content = """Player 3: {'action': 'PUT', 'marble': 'w', 'dst': 'D4', 'remove': ''}
Player 0: {'action': 'PUT', 'marble': 'g', 'dst': 'E3', 'remove': ''}
"""

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write(replay_content)
            temp_file = f.name

        try:
            loader = TranscriptLoader(temp_file)
            p1_actions, p2_actions = loader.load()

            # Invalid player numbers should be silently ignored
            assert len(p1_actions) == 0
            assert len(p2_actions) == 0
        finally:
            os.unlink(temp_file)

    def test_board_size_detection_boundary_37_48(self):
        """Test board size detection at boundary between 37 and 48 rings."""
        # Exactly G should be 37
        replay_content = """Player 1: {'action': 'PUT', 'marble': 'w', 'dst': 'G7', 'remove': ''}
"""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write(replay_content)
            temp_file = f.name

        try:
            loader = TranscriptLoader(temp_file)
            loader.load()
            assert loader.detected_rings == ZertzBoard.SMALL_BOARD_37
        finally:
            os.unlink(temp_file)

        # Exactly H should be 48
        replay_content = """Player 1: {'action': 'PUT', 'marble': 'w', 'dst': 'H8', 'remove': ''}
"""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write(replay_content)
            temp_file = f.name

        try:
            loader = TranscriptLoader(temp_file)
            loader.load()
            assert loader.detected_rings == ZertzBoard.MEDIUM_BOARD_48
        finally:
            os.unlink(temp_file)

    def test_board_size_detection_boundary_48_61(self):
        """Test board size detection at boundary between 48 and 61 rings."""
        # Exactly H should be 48
        replay_content = """Player 1: {'action': 'PUT', 'marble': 'w', 'dst': 'H1', 'remove': ''}
"""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write(replay_content)
            temp_file = f.name

        try:
            loader = TranscriptLoader(temp_file)
            loader.load()
            assert loader.detected_rings == ZertzBoard.MEDIUM_BOARD_48
        finally:
            os.unlink(temp_file)

        # I is skipped, J should trigger 61
        replay_content = """Player 1: {'action': 'PUT', 'marble': 'w', 'dst': 'J1', 'remove': ''}
"""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write(replay_content)
            temp_file = f.name

        try:
            loader = TranscriptLoader(temp_file)
            loader.load()
            assert loader.detected_rings == ZertzBoard.LARGE_BOARD_61
        finally:
            os.unlink(temp_file)

    def test_short_position_strings(self):
        """Test handling of position strings shorter than 2 characters."""
        replay_content = """Player 1: {'action': 'PUT', 'marble': 'w', 'dst': 'A', 'remove': ''}
"""

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write(replay_content)
            temp_file = f.name

        try:
            loader = TranscriptLoader(temp_file)
            p1_actions, p2_actions = loader.load()

            # Should not crash, should default to 37 rings
            assert loader.detected_rings == ZertzBoard.SMALL_BOARD_37
            # Action should still be loaded
            assert len(p1_actions) == 1
        finally:
            os.unlink(temp_file)

    def test_none_values_in_coordinate_fields(self):
        """Test handling of None values in coordinate fields."""
        replay_content = """Player 1: {'action': 'PUT', 'marble': 'w', 'dst': None, 'remove': ''}
"""

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write(replay_content)
            temp_file = f.name

        try:
            loader = TranscriptLoader(temp_file)
            p1_actions, p2_actions = loader.load()

            # Should not crash
            assert loader.detected_rings == ZertzBoard.SMALL_BOARD_37
            assert len(p1_actions) == 1
        finally:
            os.unlink(temp_file)

    def test_numeric_coordinate_values(self):
        """Test handling of numeric values in coordinate fields."""
        replay_content = """Player 1: {'action': 'PUT', 'marble': 'w', 'dst': 123, 'remove': ''}
"""

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write(replay_content)
            temp_file = f.name

        try:
            loader = TranscriptLoader(temp_file)
            p1_actions, p2_actions = loader.load()

            # Should convert to string and not crash
            assert len(p1_actions) == 1
            assert p1_actions[0]['dst'] == 123  # Preserved as-is
        finally:
            os.unlink(temp_file)

    def test_mixed_valid_and_invalid_lines(self):
        """Test file with mix of valid and invalid lines (junk lines ignored)."""
        replay_content = """Player 1: {'action': 'PUT', 'marble': 'w', 'dst': 'D4', 'remove': ''}
This is not a valid line
Player 2: {'action': 'PUT', 'marble': 'g', 'dst': 'E3', 'remove': ''}
Another invalid line: with colon
Player 1: {'action': 'PASS'}
"""

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write(replay_content)
            temp_file = f.name

        try:
            loader = TranscriptLoader(temp_file)
            # Invalid lines that don't start with "Player " are silently ignored
            p1_actions, p2_actions = loader.load()

            # Should successfully load only the valid Player lines
            assert len(p1_actions) == 2
            assert len(p2_actions) == 1
            assert p1_actions[0]['action'] == 'PUT'
            assert p1_actions[1]['action'] == 'PASS'
        finally:
            os.unlink(temp_file)

    def test_unknown_action_type(self):
        """Test handling of unknown action types."""
        replay_content = """Player 1: {'action': 'UNKNOWN', 'marble': 'w', 'dst': 'D4'}
"""

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write(replay_content)
            temp_file = f.name

        try:
            loader = TranscriptLoader(temp_file)
            p1_actions, p2_actions = loader.load()

            # Should load the action even if type is unknown
            assert len(p1_actions) == 1
            assert p1_actions[0]['action'] == 'UNKNOWN'
        finally:
            os.unlink(temp_file)

    def test_action_dict_with_extra_fields(self):
        """Test handling of action dictionaries with extra fields."""
        replay_content = """Player 1: {'action': 'PUT', 'marble': 'w', 'dst': 'D4', 'remove': '', 'extra': 'field', 'another': 123}
"""

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write(replay_content)
            temp_file = f.name

        try:
            loader = TranscriptLoader(temp_file)
            p1_actions, p2_actions = loader.load()

            # Should load successfully with extra fields
            assert len(p1_actions) == 1
            assert p1_actions[0]['extra'] == 'field'
            assert p1_actions[0]['another'] == 123
        finally:
            os.unlink(temp_file)

    def test_player_numbers_not_sequential(self):
        """Test handling of non-sequential player numbers."""
        replay_content = """Player 2: {'action': 'PUT', 'marble': 'w', 'dst': 'D4', 'remove': ''}
Player 1: {'action': 'PUT', 'marble': 'g', 'dst': 'E3', 'remove': ''}
Player 2: {'action': 'PASS'}
"""

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write(replay_content)
            temp_file = f.name

        try:
            loader = TranscriptLoader(temp_file)
            p1_actions, p2_actions = loader.load()

            # Should correctly assign actions by player number regardless of order
            assert len(p1_actions) == 1
            assert len(p2_actions) == 2
            assert p1_actions[0]['marble'] == 'g'
        finally:
            os.unlink(temp_file)

    def test_duplicate_player_actions_in_sequence(self):
        """Test multiple consecutive actions from same player."""
        replay_content = """Player 1: {'action': 'PUT', 'marble': 'w', 'dst': 'D4', 'remove': ''}
Player 1: {'action': 'PUT', 'marble': 'g', 'dst': 'E3', 'remove': ''}
Player 1: {'action': 'PUT', 'marble': 'b', 'dst': 'F2', 'remove': ''}
"""

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write(replay_content)
            temp_file = f.name

        try:
            loader = TranscriptLoader(temp_file)
            p1_actions, p2_actions = loader.load()

            # Should load all actions for player 1
            assert len(p1_actions) == 3
            assert len(p2_actions) == 0
        finally:
            os.unlink(temp_file)

    def test_coordinates_with_special_characters(self):
        """Test handling of coordinates with special characters."""
        replay_content = """Player 1: {'action': 'PUT', 'marble': 'w', 'dst': 'D@4', 'remove': ''}
"""

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write(replay_content)
            temp_file = f.name

        try:
            loader = TranscriptLoader(temp_file)
            p1_actions, p2_actions = loader.load()

            # Should load but board detection should handle gracefully
            assert len(p1_actions) == 1
            # '@' < 'G' so should default to 37
            assert loader.detected_rings == ZertzBoard.SMALL_BOARD_37
        finally:
            os.unlink(temp_file)