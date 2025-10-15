import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from controller.action_processor import ActionProcessor
from game.action_result import ActionResult


def test_process_handles_isolation_and_renderer_notifications():
    renderer = Mock()
    player = Mock()
    result = ActionResult(
        captured_marbles=[
            {"marble": "w", "pos": "A1"},
            {"marble": None, "pos": "B2"},
        ]
    )

    processor = ActionProcessor(renderer)
    processor.process(player, result, delay_time=0.5)

    player.add_capture.assert_called_once_with("w")
    renderer.show_isolated_removal.assert_any_call(player, "A1", "w", 0.5)
    renderer.show_isolated_removal.assert_any_call(player, "B2", None, 0.5)
    assert renderer.show_isolated_removal.call_count == 2


def test_process_handles_standard_capture_without_renderer():
    player = Mock()
    processor = ActionProcessor(None)
    result = ActionResult(captured_marbles="g")

    processor.process(player, result, delay_time=0.25)

    player.add_capture.assert_called_once_with("g")


def test_process_ignores_empty_results():
    player = Mock()
    renderer = Mock()
    processor = ActionProcessor(renderer)

    processor.process(player, None, delay_time=1.0)

    player.add_capture.assert_not_called()
    renderer.show_isolated_removal.assert_not_called()
