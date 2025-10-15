"""Replay loader for Zertz 3D game.

Handles loading and parsing replay files, including board size detection
and variant detection.
"""

import ast
from typing import Callable
from game.zertz_board import ZertzBoard
from game.zertz_game import (
    BLITZ_MARBLES,
    BLITZ_WIN_CONDITIONS,
    STANDARD_MARBLES,
    STANDARD_WIN_CONDITIONS,
)


class ReplayLoader:
    """Loads and parses replay files for Zertz games."""

    def __init__(
        self,
        replay_file,
        blitz=False,
        status_reporter: Callable[[str], None] | None = None,
    ):
        """Initialize the replay loader.

        Args:
            replay_file: Path to the replay file
            blitz: Whether blitz mode is enabled via command line
        """
        self.replay_file = replay_file
        self.blitz = blitz
        self.detected_rings = None
        self.detected_blitz = False
        self.marbles = None
        self.win_condition = None
        self._status_reporter: Callable[[str], None] | None = status_reporter

    def load(self):
        """Load replay actions from the file.

        Returns:
            Tuple of (player1_actions, player2_actions)

        Side effects:
            Sets self.detected_rings, self.detected_blitz, self.marbles,
            and self.win_condition based on file contents.
        """
        self._report(f"Loading replay from: {self.replay_file}")
        player1_actions = []
        player2_actions = []
        all_actions = []

        with open(self.replay_file, "r") as f:
            for line in f:
                line = line.strip()

                # Check for variant in comment headers
                if line.startswith("# Variant:"):
                    variant = line.split(":", 1)[1].strip().lower()
                    if variant == "blitz":
                        self.detected_blitz = True

                if not line or line.startswith("#"):
                    continue

                # Parse line format: "Player N: {'action': 'PUT', ...}"
                if line.startswith("Player "):
                    parts = line.split(": ", 1)
                    player_num = int(parts[0].split()[1])
                    action_dict = ast.literal_eval(parts[1])

                    all_actions.append(action_dict)
                    if player_num == 1:
                        player1_actions.append(action_dict)
                    elif player_num == 2:
                        player2_actions.append(action_dict)

        # Detect board size from coordinates
        self.detected_rings = self._detect_board_size(all_actions)
        self._report(f"Detected board size: {self.detected_rings} rings")

        # Determine final variant (file takes precedence over command line)
        if self.detected_blitz:
            self._report("Detected blitz variant from replay file")
            if self.blitz:
                self._report("  (--blitz flag also specified)")
            else:
                self._report("  (automatically enabling blitz mode)")
            self.blitz = True
        elif self.blitz:
            self._report(
                "Warning: --blitz flag specified but replay file is standard mode"
            )
            self._report("         Using blitz rules anyway")

        # Always set marbles and win_condition based on final variant
        if self.blitz:
            self.marbles = BLITZ_MARBLES
            self.win_condition = BLITZ_WIN_CONDITIONS
        else:
            self.marbles = STANDARD_MARBLES
            self.win_condition = STANDARD_WIN_CONDITIONS

        self._report(f"Loaded {len(player1_actions)} actions for Player 1")
        self._report(f"Loaded {len(player2_actions)} actions for Player 2")
        return player1_actions, player2_actions

    def _detect_board_size(self, all_actions):
        """Detect board size by finding the maximum letter coordinate used.

        Args:
            all_actions: List of all action dictionaries

        Returns:
            Number of rings (37, 48, or 61)
        """
        max_letter = "A"

        for action in all_actions:
            # Check all position fields that might contain ring coordinates
            for key in ["dst", "src", "remove", "cap"]:
                if key in action and action[key]:
                    pos = str(action[key])
                    if len(pos) >= 2:
                        letter = pos[0].upper()
                        if letter > max_letter:
                            max_letter = letter

        # Detect board size from max letter (number of columns)
        # For hexagonal boards:
        #   A-G (7 letters) = 37 rings
        #   A-H (8 letters) = 48 rings
        #   A-J (9 letters, skipping I) = 61 rings
        if max_letter <= "G":
            return ZertzBoard.SMALL_BOARD_37
        elif max_letter <= "H":
            return ZertzBoard.MEDIUM_BOARD_48
        else:
            # J or beyond = 61 ring board
            return ZertzBoard.LARGE_BOARD_61

    def set_status_reporter(self, reporter: Callable[[str], None] | None) -> None:
        """Update status reporter after initialization."""
        self._status_reporter = reporter

    def _report(self, message: str | None) -> None:
        if message is None:
            return
        if self._status_reporter is not None:
            self._status_reporter(message)
        else:
            print(message)
