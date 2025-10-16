"""
Unit tests for ActionResult.

Tests the ActionResult value object that encapsulates action execution results.
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from game.action_result import ActionResult


class TestActionResult:
    """Test ActionResult behavior."""

    def test_action_result_no_isolation(self):
        """Test ActionResult with no isolation."""
        result = ActionResult(captured_marbles="w")

        assert not result.is_isolation(), "Single capture should not be isolation"
        assert result.captured_marbles == "w"

    def test_action_result_with_isolation(self):
        """Test ActionResult with isolation captures."""
        result = ActionResult(
            captured_marbles=[
                {"marble": "w", "pos": "A1"},
                {"marble": "b", "pos": "B2"},
            ]
        )

        assert result.is_isolation(), "Multiple captures should indicate isolation"
        assert len(result.captured_marbles) == 2

    def test_action_result_no_captures(self):
        """Test ActionResult with no captures."""
        result = ActionResult(captured_marbles=None)

        assert not result.is_isolation(), "No captures should not be isolation"
        assert result.captured_marbles is None

    def test_action_result_empty_isolation_list(self):
        """Test ActionResult with empty isolation list."""
        result = ActionResult(captured_marbles=[])

        # Empty list should still be treated as isolation (even though it's degenerate)
        assert result.is_isolation(), "Empty list should indicate isolation structure"
        assert len(result.captured_marbles) == 0

    def test_action_result_single_isolation_capture(self):
        """Test ActionResult with single item in isolation list."""
        result = ActionResult(
            captured_marbles=[{"marble": "w", "pos": "A1"}]
        )

        assert result.is_isolation(), "List structure should indicate isolation"
        assert len(result.captured_marbles) == 1
        assert result.captured_marbles[0]["marble"] == "w"
        assert result.captured_marbles[0]["pos"] == "A1"
