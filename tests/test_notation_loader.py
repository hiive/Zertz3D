"""Tests for NotationLoader."""

import tempfile
import os
import pytest

from game.loaders import NotationLoader
from game.zertz_board import ZertzBoard
from game.zertz_game import (
    STANDARD_MARBLES,
    BLITZ_MARBLES,
    STANDARD_WIN_CONDITIONS,
    BLITZ_WIN_CONDITIONS,
)


class TestNotationLoader:
    """Test suite for NotationLoader class."""

    def test_basic_37_ring_notation(self):
        """Test loading a basic 37-ring game notation."""
        notation_content = """37
Bd7,g4
Ge1,e6
Wd3,f5
x c2We3
-
"""

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write(notation_content)
            temp_file = f.name

        try:
            loader = NotationLoader(temp_file)
            p1_actions, p2_actions = loader.load()

            assert loader.detected_rings == ZertzBoard.SMALL_BOARD_37
            assert not loader.blitz
            assert loader.marbles == STANDARD_MARBLES
            assert loader.win_condition == STANDARD_WIN_CONDITIONS

            assert len(p1_actions) == 3
            assert len(p2_actions) == 2

            # Check first action (P1 move 1: Bd7,g4)
            assert p1_actions[0]["action"] == "PUT"
            assert p1_actions[0]["marble"] == "b"
            assert p1_actions[0]["dst"] == "d7"

            # Check pass action (P1 move 5: -)
            assert p1_actions[2]["action"] == "PASS"

            # Check capture action (P2 move 4: x c2We3)
            assert p2_actions[1]["action"] == "CAP"
            assert p2_actions[1]["src"] == "c2"
        finally:
            os.unlink(temp_file)

    def test_48_ring_board_detection(self):
        """Test detection of 48-ring board."""
        notation_content = """48
Wd4
Ge3
"""

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write(notation_content)
            temp_file = f.name

        try:
            loader = NotationLoader(temp_file)
            p1_actions, p2_actions = loader.load()

            assert loader.detected_rings == ZertzBoard.MEDIUM_BOARD_48
        finally:
            os.unlink(temp_file)

    def test_61_ring_board_detection(self):
        """Test detection of 61-ring board."""
        notation_content = """61
Wj1
Ga9
"""

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write(notation_content)
            temp_file = f.name

        try:
            loader = NotationLoader(temp_file)
            p1_actions, p2_actions = loader.load()

            assert loader.detected_rings == ZertzBoard.LARGE_BOARD_61
        finally:
            os.unlink(temp_file)

    def test_blitz_variant_from_file(self):
        """Test blitz variant detection from file header."""
        notation_content = """37 Blitz
Wd4
Ge3
"""

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write(notation_content)
            temp_file = f.name

        try:
            loader = NotationLoader(temp_file)
            p1_actions, p2_actions = loader.load()

            assert loader.blitz
            assert loader.marbles == BLITZ_MARBLES
            assert loader.win_condition == BLITZ_WIN_CONDITIONS
        finally:
            os.unlink(temp_file)

    def test_player_alternation(self):
        """Test that actions alternate between players."""
        notation_content = """37
Wd4
Ge3
Bd2
Wf1
"""

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write(notation_content)
            temp_file = f.name

        try:
            loader = NotationLoader(temp_file)
            p1_actions, p2_actions = loader.load()

            # P1 gets moves 1, 3 (Wd4, Bd2)
            assert len(p1_actions) == 2
            assert p1_actions[0]["marble"] == "w"
            assert p1_actions[1]["marble"] == "b"

            # P2 gets moves 2, 4 (Ge3, Wf1)
            assert len(p2_actions) == 2
            assert p2_actions[0]["marble"] == "g"
            assert p2_actions[1]["marble"] == "w"
        finally:
            os.unlink(temp_file)

    def test_empty_lines_ignored(self):
        """Test that empty lines are ignored."""
        notation_content = """37

Wd4

Ge3

"""

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write(notation_content)
            temp_file = f.name

        try:
            loader = NotationLoader(temp_file)
            p1_actions, p2_actions = loader.load()

            assert len(p1_actions) == 1
            assert len(p2_actions) == 1
        finally:
            os.unlink(temp_file)

    def test_placement_with_isolation(self):
        """Test placement with isolation captures."""
        notation_content = """37
Wf2,d4 x Wa1Gd5
Ge3
"""

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write(notation_content)
            temp_file = f.name

        try:
            loader = NotationLoader(temp_file)
            p1_actions, p2_actions = loader.load()

            # Isolation part should be stripped
            assert p1_actions[0]["marble"] == "w"
            assert p1_actions[0]["dst"] == "f2"
            assert p1_actions[0]["remove"] == "d4"
        finally:
            os.unlink(temp_file)

    def test_capture_actions(self):
        """Test capture action parsing."""
        notation_content = """37
Wd4
x c2We3
Ge1
x e3Gg1
"""

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write(notation_content)
            temp_file = f.name

        try:
            loader = NotationLoader(temp_file)
            p1_actions, p2_actions = loader.load()

            # P1: Wd4 (move 1), Ge1 (move 3)
            assert p1_actions[0]["action"] == "PUT"
            assert p1_actions[1]["action"] == "PUT"
            assert p1_actions[1]["marble"] == "g"

            # P2: x c2We3 (move 2), x e3Gg1 (move 4)
            assert p2_actions[0]["action"] == "CAP"
            assert p2_actions[0]["src"] == "c2"
            assert p2_actions[1]["action"] == "CAP"
            assert p2_actions[1]["src"] == "e3"
            assert p2_actions[1]["capture"] == "g"
        finally:
            os.unlink(temp_file)

    def test_pass_actions(self):
        """Test pass action parsing."""
        notation_content = """37
Wd4
-
Ge3
-
"""

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write(notation_content)
            temp_file = f.name

        try:
            loader = NotationLoader(temp_file)
            p1_actions, p2_actions = loader.load()

            # P1: Wd4 (move 1), Ge3 (move 3)
            assert p1_actions[0]["action"] == "PUT"
            assert p1_actions[1]["action"] == "PUT"

            # P2: - (move 2), - (move 4)
            assert p2_actions[0]["action"] == "PASS"
            assert p2_actions[1]["action"] == "PASS"
        finally:
            os.unlink(temp_file)

    def test_invalid_notation_skipped_with_warning(self):
        """Test that invalid notation lines are skipped."""
        notation_content = """37
Wd4
INVALID LINE
Ge3
"""

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write(notation_content)
            temp_file = f.name

        try:
            messages = []

            def reporter(msg):
                messages.append(msg)

            loader = NotationLoader(temp_file, status_reporter=reporter)
            p1_actions, p2_actions = loader.load()

            # Should have warning about invalid line
            assert any("Warning" in msg and "invalid notation" in msg.lower() for msg in messages)

            # Should still load valid actions (skipping invalid)
            assert len(p1_actions) == 1
            assert len(p2_actions) == 1
        finally:
            os.unlink(temp_file)

    def test_empty_file(self):
        """Test handling of empty file."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("")
            temp_file = f.name

        try:
            loader = NotationLoader(temp_file)
            p1_actions, p2_actions = loader.load()

            assert len(p1_actions) == 0
            assert len(p2_actions) == 0
            assert loader.detected_rings == ZertzBoard.SMALL_BOARD_37
        finally:
            os.unlink(temp_file)

    def test_header_only_file(self):
        """Test file with only header."""
        notation_content = """37
"""

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write(notation_content)
            temp_file = f.name

        try:
            loader = NotationLoader(temp_file)
            p1_actions, p2_actions = loader.load()

            assert len(p1_actions) == 0
            assert len(p2_actions) == 0
            assert loader.detected_rings == ZertzBoard.SMALL_BOARD_37
        finally:
            os.unlink(temp_file)

    def test_invalid_board_size_defaults_to_37(self):
        """Test that invalid board size defaults to 37."""
        notation_content = """99
Wd4
"""

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write(notation_content)
            temp_file = f.name

        try:
            messages = []

            def reporter(msg):
                messages.append(msg)

            loader = NotationLoader(temp_file, status_reporter=reporter)
            p1_actions, p2_actions = loader.load()

            # Should warn about unknown size
            assert any("Unknown board size" in msg for msg in messages)
            assert loader.detected_rings == ZertzBoard.SMALL_BOARD_37
        finally:
            os.unlink(temp_file)

    def test_non_numeric_board_size_defaults_to_37(self):
        """Test that non-numeric board size defaults to 37."""
        notation_content = """ABC
Wd4
"""

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write(notation_content)
            temp_file = f.name

        try:
            messages = []

            def reporter(msg):
                messages.append(msg)

            loader = NotationLoader(temp_file, status_reporter=reporter)
            p1_actions, p2_actions = loader.load()

            assert any("Invalid board size" in msg for msg in messages)
            assert loader.detected_rings == ZertzBoard.SMALL_BOARD_37
        finally:
            os.unlink(temp_file)

    def test_variant_case_insensitive(self):
        """Test that variant detection is case insensitive."""
        notation_content = """37 BLITZ
Wd4
"""

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write(notation_content)
            temp_file = f.name

        try:
            loader = NotationLoader(temp_file)
            p1_actions, p2_actions = loader.load()

            assert loader.blitz
        finally:
            os.unlink(temp_file)

    def test_status_reporter_callback(self):
        """Test that status reporter callback is called."""
        notation_content = """37
Wd4
"""

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write(notation_content)
            temp_file = f.name

        try:
            messages = []

            def reporter(msg):
                messages.append(msg)

            loader = NotationLoader(temp_file, status_reporter=reporter)
            p1_actions, p2_actions = loader.load()

            assert len(messages) > 0
            assert any("Loading notation" in msg for msg in messages)
            assert any("Detected board size" in msg for msg in messages)
        finally:
            os.unlink(temp_file)

    def test_set_status_reporter_after_init(self):
        """Test setting status reporter after initialization."""
        notation_content = """37
Wd4
"""

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write(notation_content)
            temp_file = f.name

        try:
            messages = []

            def reporter(msg):
                messages.append(msg)

            loader = NotationLoader(temp_file)
            loader.set_status_reporter(reporter)
            p1_actions, p2_actions = loader.load()

            assert len(messages) > 0
        finally:
            os.unlink(temp_file)

    def test_file_not_found(self):
        """Test handling of non-existent file."""
        with pytest.raises(FileNotFoundError):
            loader = NotationLoader("/nonexistent/path/to/file.txt")
            loader.load()