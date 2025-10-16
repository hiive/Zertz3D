"""Tests for NotationFormatter."""

import pytest

from game.formatters import NotationFormatter
from game.action_result import ActionResult


class TestNotationFormatter:
    """Test suite for NotationFormatter class."""

    # === action_to_notation tests ===

    def test_pass_action_to_notation(self):
        """Test PASS action conversion."""
        action_dict = {"action": "PASS"}
        notation = NotationFormatter.action_to_notation(action_dict)
        assert notation == "-"

    def test_placement_no_removal_to_notation(self):
        """Test PUT action without removal."""
        action_dict = {"action": "PUT", "marble": "w", "dst": "D4", "remove": ""}
        notation = NotationFormatter.action_to_notation(action_dict)
        assert notation == "Wd4"

    def test_placement_with_removal_to_notation(self):
        """Test PUT action with removal."""
        action_dict = {"action": "PUT", "marble": "b", "dst": "D7", "remove": "G4"}
        notation = NotationFormatter.action_to_notation(action_dict)
        assert notation == "Bd7,g4"

    def test_placement_case_conversion(self):
        """Test that marble color is uppercase and positions are lowercase."""
        action_dict = {"action": "PUT", "marble": "g", "dst": "E3", "remove": "A1"}
        notation = NotationFormatter.action_to_notation(action_dict)
        assert notation == "Ge3,a1"

    def test_placement_with_isolation_capture(self):
        """Test PUT action with isolation capture."""
        action_dict = {"action": "PUT", "marble": "w", "dst": "F2", "remove": "D4"}

        # Create ActionResult with isolation captures
        captured_marbles = [
            {"marble": "w", "pos": "A1"},
            {"marble": "g", "pos": "D5"},
        ]
        action_result = ActionResult(captured_marbles=captured_marbles)

        notation = NotationFormatter.action_to_notation(action_dict, action_result)
        assert notation == "Wf2,d4 x Wa1Gd5"

    def test_placement_isolation_only_includes_marbles(self):
        """Test that isolation only includes entries with marbles."""
        action_dict = {"action": "PUT", "marble": "b", "dst": "C3", "remove": ""}

        # Captured marbles with None marble (vacant ring isolation)
        captured_marbles = [
            {"marble": "w", "pos": "B2"},
            {"marble": None, "pos": "C4"},  # Vacant ring, should be skipped
        ]
        action_result = ActionResult(captured_marbles=captured_marbles)

        notation = NotationFormatter.action_to_notation(action_dict, action_result)
        assert notation == "Bc3 x Wb2"

    def test_capture_action_to_notation(self):
        """Test CAP action conversion."""
        action_dict = {
            "action": "CAP",
            "marble": "g",
            "src": "C2",
            "dst": "E3",
            "capture": "w",
            "cap": "D2",
        }
        notation = NotationFormatter.action_to_notation(action_dict)
        assert notation == "x c2We3"

    def test_capture_case_conversion(self):
        """Test capture notation case conversion."""
        action_dict = {
            "action": "CAP",
            "marble": "b",
            "src": "E3",
            "dst": "G1",
            "capture": "g",
            "cap": "F2",
        }
        notation = NotationFormatter.action_to_notation(action_dict)
        assert notation == "x e3Gg1"

    # === notation_to_action_dict tests ===

    def test_parse_pass_notation(self):
        """Test parsing PASS notation."""
        action_dict = NotationFormatter.notation_to_action_dict("-")
        assert action_dict == {"action": "PASS"}

    def test_parse_placement_no_removal(self):
        """Test parsing placement without removal."""
        action_dict = NotationFormatter.notation_to_action_dict("Wd4")
        assert action_dict == {
            "action": "PUT",
            "marble": "w",
            "dst": "d4",
            "remove": "",
        }

    def test_parse_placement_with_removal(self):
        """Test parsing placement with removal."""
        action_dict = NotationFormatter.notation_to_action_dict("Bd7,g4")
        assert action_dict == {
            "action": "PUT",
            "marble": "b",
            "dst": "d7",
            "remove": "g4",
        }

    def test_parse_placement_all_colors(self):
        """Test parsing placements for all marble colors."""
        for color in ["W", "G", "B"]:
            action_dict = NotationFormatter.notation_to_action_dict(f"{color}e3")
            assert action_dict["marble"] == color.lower()

    def test_parse_placement_with_isolation_strips_isolation(self):
        """Test that isolation captures are stripped from placement notation."""
        action_dict = NotationFormatter.notation_to_action_dict("Wf2,d4 x Wa1Gd5")
        # Isolation part should be ignored
        assert action_dict == {
            "action": "PUT",
            "marble": "w",
            "dst": "f2",
            "remove": "d4",
        }

    def test_parse_capture_notation(self):
        """Test parsing capture notation."""
        action_dict = NotationFormatter.notation_to_action_dict("x c2We3")
        assert action_dict == {
            "action": "CAP",
            "src": "c2",
            "dst": "e3",
            "capture": "w",
            "cap": "d2",  # Calculated from src and dst
        }

    def test_parse_capture_all_colors(self):
        """Test parsing captures for all marble colors."""
        for color in ["W", "G", "B"]:
            action_dict = NotationFormatter.notation_to_action_dict(f"x a1{color}c3")
            assert action_dict["capture"] == color.lower()

    def test_parse_capture_case_insensitive_positions(self):
        """Test that capture positions are case insensitive."""
        action_dict = NotationFormatter.notation_to_action_dict("x C2We3")
        assert action_dict["src"] == "C2"  # Preserved as-is
        assert action_dict["dst"] == "e3"

    def test_parse_placement_61_ring_coordinates(self):
        """Test parsing placement with J coordinate (61-ring board)."""
        action_dict = NotationFormatter.notation_to_action_dict("Wj1,a9")
        assert action_dict == {
            "action": "PUT",
            "marble": "w",
            "dst": "j1",
            "remove": "a9",
        }

    def test_parse_invalid_placement_raises_error(self):
        """Test that invalid placement notation raises ValueError."""
        with pytest.raises(ValueError, match="Invalid placement notation"):
            NotationFormatter.notation_to_action_dict("XYZ")

        with pytest.raises(ValueError, match="Invalid placement notation"):
            NotationFormatter.notation_to_action_dict("W")  # Missing position

        with pytest.raises(ValueError, match="Invalid placement notation"):
            NotationFormatter.notation_to_action_dict("Wd")  # Missing number

    def test_parse_invalid_capture_raises_error(self):
        """Test that invalid capture notation raises ValueError."""
        with pytest.raises(ValueError, match="Invalid capture notation"):
            NotationFormatter.notation_to_action_dict("x abc")

        with pytest.raises(ValueError, match="Invalid capture notation"):
            NotationFormatter.notation_to_action_dict("x a1")  # Missing color and dst

        with pytest.raises(ValueError, match="Invalid capture notation"):
            NotationFormatter.notation_to_action_dict("x a1Xb2")  # Invalid color

    def test_parse_whitespace_handling(self):
        """Test that leading/trailing whitespace is handled."""
        action_dict = NotationFormatter.notation_to_action_dict("  Wd4  ")
        assert action_dict["marble"] == "w"

        action_dict = NotationFormatter.notation_to_action_dict("  x c2We3  ")
        assert action_dict["action"] == "CAP"

    def test_roundtrip_placement_no_removal(self):
        """Test roundtrip: action -> notation -> action."""
        original = {"action": "PUT", "marble": "w", "dst": "D4", "remove": ""}
        notation = NotationFormatter.action_to_notation(original)
        parsed = NotationFormatter.notation_to_action_dict(notation)

        assert parsed["action"] == original["action"]
        assert parsed["marble"] == original["marble"]
        assert parsed["dst"].upper() == original["dst"].upper()
        assert parsed["remove"] == original["remove"]

    def test_roundtrip_placement_with_removal(self):
        """Test roundtrip with removal."""
        original = {"action": "PUT", "marble": "b", "dst": "D7", "remove": "G4"}
        notation = NotationFormatter.action_to_notation(original)
        parsed = NotationFormatter.notation_to_action_dict(notation)

        assert parsed["action"] == original["action"]
        assert parsed["marble"] == original["marble"]
        assert parsed["dst"].upper() == original["dst"].upper()
        assert parsed["remove"].upper() == original["remove"].upper()

    def test_roundtrip_capture(self):
        """Test roundtrip for capture."""
        original = {
            "action": "CAP",
            "marble": "g",
            "src": "C2",
            "dst": "E3",
            "capture": "w",
            "cap": "D2",
        }
        notation = NotationFormatter.action_to_notation(original)
        parsed = NotationFormatter.notation_to_action_dict(notation)

        assert parsed["action"] == original["action"]
        assert parsed["src"].upper() == original["src"].upper()
        assert parsed["dst"].upper() == original["dst"].upper()
        assert parsed["capture"] == original["capture"]
        # Note: cap position is calculated from src/dst
        assert parsed["cap"].upper() == original["cap"].upper()

    def test_roundtrip_pass(self):
        """Test roundtrip for pass."""
        original = {"action": "PASS"}
        notation = NotationFormatter.action_to_notation(original)
        parsed = NotationFormatter.notation_to_action_dict(notation)

        assert parsed == original