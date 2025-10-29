"""Loader for official Zèrtz notation files.

Parses notation files (e.g., zertzlog_1759986153_notation.txt) into action dictionaries
that can be used with ReplayZertzPlayer.
"""

from typing import Callable

from game.constants import (
    STANDARD_MARBLES,
    BLITZ_MARBLES,
    STANDARD_WIN_CONDITIONS,
    BLITZ_WIN_CONDITIONS,
)
from game.formatters import NotationFormatter
from game.zertz_board import ZertzBoard


class NotationLoader:
    """Loads and parses official Zèrtz notation files."""

    def __init__(
        self,
        filename,
        status_reporter: Callable[[str], None] | None = None,
    ):
        """Initialize notation loader.

        Args:
            filename: Path to notation file
            status_reporter: Optional callback for status messages
        """
        self.filename = filename
        self._status_reporter = status_reporter

        # Detected values (set after load())
        self.detected_rings = ZertzBoard.SMALL_BOARD_37
        self.blitz = False
        self.marbles = STANDARD_MARBLES
        self.win_condition = STANDARD_WIN_CONDITIONS
        self.player1_name: str | None = None
        self.player2_name: str | None = None

    def set_status_reporter(self, reporter: Callable[[str], None] | None) -> None:
        """Set or update the status reporter callback."""
        self._status_reporter = reporter

    def load(self) -> tuple[list[dict], list[dict]]:
        """Load and parse notation file.

        Returns:
            Tuple of (player1_actions, player2_actions)
            Each action is a dict with keys: action, marble, dst, remove (for PUT)
                                             or: action, marble, src, dst, capture, cap (for CAP)
        """
        self._report(f"Loading notation from: {self.filename}")

        with open(self.filename, "r") as f:
            lines = f.readlines()

        if not lines:
            self._report("Empty notation file")
            return [], []

        # Parse header (first line: board size + optional variant)
        header = lines[0].strip()
        self._parse_header(header)

        # Parse moves (remaining lines)
        player1_actions = []
        player2_actions = []
        current_player = 1

        for line_num, line in enumerate(lines[1:], start=2):
            line = line.strip()
            if not line:
                continue

            # Parse player name comments
            if line.startswith("# Player 1:"):
                self.player1_name = line.split(":", 1)[1].strip()
                continue
            elif line.startswith("# Player 2:"):
                self.player2_name = line.split(":", 1)[1].strip()
                continue
            elif line.startswith("#"):
                # Other comments - skip
                continue

            try:
                # Parse notation to action dict
                action_dict = NotationFormatter.notation_to_action_dict(line)

                # Add to appropriate player's action list
                if current_player == 1:
                    player1_actions.append(action_dict)
                else:
                    player2_actions.append(action_dict)

                # Alternate players
                current_player = 2 if current_player == 1 else 1

            except ValueError as e:
                self._report(f"Warning: Skipping invalid notation on line {line_num}: {line} ({e})")
                continue

        self._report(f"Loaded {len(player1_actions)} actions for Player 1")
        self._report(f"Loaded {len(player2_actions)} actions for Player 2")
        self._report(f"Detected board size: {self.detected_rings} rings")

        return player1_actions, player2_actions

    def _parse_header(self, header: str) -> None:
        """Parse header line to detect board size and variant.

        Header format: "37" or "37 Blitz" or "48" or "61 Blitz"

        Args:
            header: First line of notation file
        """
        self.blitz = False
        parts = header.split()
        if not parts:
            self.detected_rings = ZertzBoard.SMALL_BOARD_37
            return

        # First part is board size
        try:
            rings = int(parts[0])
            if rings == 37:
                self.detected_rings = ZertzBoard.SMALL_BOARD_37
            elif rings == 48:
                self.detected_rings = ZertzBoard.MEDIUM_BOARD_48
            elif rings == 61:
                self.detected_rings = ZertzBoard.LARGE_BOARD_61
            else:
                self._report(f"Warning: Unknown board size {rings}, defaulting to 37")
                self.detected_rings = ZertzBoard.SMALL_BOARD_37
        except ValueError:
            self._report("Warning: Invalid board size in header, defaulting to 37")
            self.detected_rings = ZertzBoard.SMALL_BOARD_37

        # Check for variant in remaining parts
        if len(parts) > 1:
            variant = " ".join(parts[1:]).lower()
            self.blitz = "blitz" in variant

        # Set blitz mode based on file detection only
        if self.blitz:
            self.marbles = BLITZ_MARBLES
            self.win_condition = BLITZ_WIN_CONDITIONS
        else:
            self.marbles = STANDARD_MARBLES
            self.win_condition = STANDARD_WIN_CONDITIONS

    def _report(self, message: str | None) -> None:
        """Report status message via callback or print."""
        if message is None:
            return
        if self._status_reporter is not None:
            self._status_reporter(message)
        else:
            print(message)