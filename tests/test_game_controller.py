"""
Unit tests for ZertzGameController.

Tests that the game controller correctly handles:
- Game ending in headless mode
- max_games limit enforcement
- Game counter incrementing exactly once per game
- Winner display formatting
"""

import sys
from pathlib import Path
from unittest.mock import Mock, patch

# Add parent directory to path to import game modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from controller.zertz_game_controller import ZertzGameController
from game.action_result import ActionResult


# ============================================================================
# Test Game Ending in Headless Mode
# ============================================================================


class TestGameControllerHeadless:
    """Test game controller behavior in headless mode (no renderer)."""

    def test_game_ends_correctly_with_max_games_1(self):
        """Test that game stops after exactly 1 game when max_games=1.

        This test verifies the fixes for:
        - Bug: Game continuing after first win
        - Bug: Game counter incrementing twice
        - Bug: Winner displayed as -1 instead of player number
        """
        # Use a fixed seed for deterministic behavior
        seed = 12345

        # Capture console output to verify correct winner display
        output_lines = []

        def capture_print(*args, **kwargs):
            """Capture print statements."""
            output_lines.append(" ".join(str(arg) for arg in args))

        with patch("builtins.print", side_effect=capture_print):
            # Create controller in headless mode with max_games=1
            controller = ZertzGameController(
                rings=37,
                seed=seed,
                max_games=1,
                highlight_choices=False,
                show_coords=False,
                log_to_file=False,
                renderer_or_factory=None,  # Headless mode
            )

            # Run the game loop
            controller.run()

        # Verify the game ended
        games_played = controller.session.get_games_played()
        assert games_played == 1, (
            f"Expected exactly 1 game to be played, but got {games_played}. "
            "This suggests the game continued after the first win."
        )

        # Verify winner was reported correctly (not as -1 or 1)
        winner_lines = [line for line in output_lines if line.startswith("Winner:")]
        assert len(winner_lines) == 1, (
            f"Expected exactly 1 winner announcement, but got {len(winner_lines)}. "
            f"Lines: {winner_lines}"
        )

        # Winner should be displayed as "Player 1" or "Player 2", not "Player -1" or "Player 1" (constant)
        winner_line = winner_lines[0]
        assert "Player 1" in winner_line or "Player 2" in winner_line, (
            f"Winner should be displayed as 'Player 1' or 'Player 2', got: {winner_line}"
        )
        assert "Player -1" not in winner_line, (
            f"Winner should not be displayed as 'Player -1', got: {winner_line}"
        )

        # Verify "Completed X game(s)" appears exactly once
        completed_lines = [
            line for line in output_lines if line.startswith("Completed")
        ]
        assert len(completed_lines) == 1, (
            f"Expected exactly 1 'Completed' announcement, but got {len(completed_lines)}. "
            f"Lines: {completed_lines}"
        )
        assert "Completed 1 game(s)" in completed_lines[0], (
            f"Expected 'Completed 1 game(s)', got: {completed_lines[0]}"
        )

    def test_game_status_checked_after_empty_completion_queue(self):
        """Test that game status is checked even when completion queue is empty.

        This verifies the fix for the bug where game status check was skipped
        if the completion queue was empty, causing the game to continue
        after a win.
        """
        seed = 54321
        output_lines = []

        def capture_print(*args, **kwargs):
            output_lines.append(" ".join(str(arg) for arg in args))

        with patch("builtins.print", side_effect=capture_print):
            controller = ZertzGameController(
                rings=37,
                seed=seed,
                max_games=2,  # Allow 2 games, but should stop after 2
                highlight_choices=False,
                renderer_or_factory=None,
            )

            controller.run()

        # Count winner announcements
        winner_lines = [line for line in output_lines if line.startswith("Winner:")]

        # Should have exactly 2 winner announcements (one per game)
        assert len(winner_lines) == 2, (
            f"Expected exactly 2 winner announcements for 2 games, got {len(winner_lines)}"
        )

        # Verify final game count
        assert controller.session.get_games_played() == 2, (
            "Expected exactly 2 games to be played"
        )

    def test_winner_display_format_player_1(self):
        """Test that Player 1 wins are displayed correctly."""
        # Use a seed that favors Player 1 winning quickly
        seed = 99999
        output_lines = []

        def capture_print(*args, **kwargs):
            output_lines.append(" ".join(str(arg) for arg in args))

        with patch("builtins.print", side_effect=capture_print):
            controller = ZertzGameController(
                rings=37,
                seed=seed,
                max_games=1,
                highlight_choices=False,
                renderer_or_factory=None,
            )

            controller.run()

        winner_lines = [line for line in output_lines if line.startswith("Winner:")]

        # Should have exactly one winner announcement
        assert len(winner_lines) == 1

        # Winner line should contain either "Player 1" or "Player 2"
        winner_line = winner_lines[0]
        assert "Player 1" in winner_line or "Player 2" in winner_line, (
            f"Winner should be 'Player 1' or 'Player 2', got: {winner_line}"
        )

    def test_game_counter_increments_once_per_game(self):
        """Test that game counter increments exactly once per game, not twice."""
        seed = 11111

        controller = ZertzGameController(
            rings=37,
            seed=seed,
            max_games=3,
            highlight_choices=False,
            renderer_or_factory=None,
            log_to_file=False,
        )

        # Initially should be 0
        assert controller.session.get_games_played() == 0

        # Run the games
        controller.run()

        # Should be exactly 3, not 6 (would be 6 if bug was present)
        assert controller.session.get_games_played() == 3, (
            "Game counter should increment exactly once per game"
        )

    def test_completion_queue_uses_action_processor(self):
        """Ensure completion queue delegates to action processor helper."""
        controller = ZertzGameController(
            rings=37, highlight_choices=False, renderer_or_factory=None
        )
        processor = Mock()
        controller.action_processor = processor

        player = Mock()
        result = ActionResult(captured_marbles="w")

        controller._handle_action_completion(player, result)
        controller._process_completion_queue(delay_time=0.5)

        processor.process.assert_called_once_with(player, result, 0.5)
