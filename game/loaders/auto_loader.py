"""Auto-detecting replay loader.

Automatically detects the format of a replay file (SGF, notation, or transcript)
and delegates to the appropriate loader.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from .base_loader import ReplayLoader
from .notation_loader import NotationLoader
from .sgf_loader import SGFLoader
from .transcript_loader import TranscriptLoader


class AutoSelectLoader:
    """Loader that automatically detects file format and delegates to appropriate loader.

    Implements the same interface as individual loaders (ReplayLoader protocol)
    so it can be used as a drop-in replacement.

    Detection logic:
    - Files ending in .sgf -> SGFLoader
    - Files starting with '#' or 'Player' -> TranscriptLoader
    - Files starting with a digit -> NotationLoader
    - Default: TranscriptLoader
    """

    def __init__(
        self,
        filename: str | Path,
        status_reporter: Callable[[str], None] | None = None,
    ):
        """Initialize auto-selecting loader.

        Args:
            filename: Path to replay file
            status_reporter: Optional callback for status messages
        """
        self.filename = Path(filename)
        self._status_reporter = status_reporter
        self._delegate: ReplayLoader | None = None

    @staticmethod
    def detect_file_format(filepath: str | Path) -> str:
        """Detect whether a file is transcript, notation, or SGF format.

        Args:
            filepath: Path to the file to detect

        Returns:
            "sgf", "notation", or "transcript"
        """
        filepath = Path(filepath)

        # Check file extension first
        if filepath.suffix.lower() == ".sgf":
            return "sgf"

        # Read file content to determine format
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:  # Skip empty lines
                    continue

                # Transcript files start with '#' (comments) or 'Player'
                if line.startswith('#') or line.startswith('Player'):
                    return "transcript"

                # Notation files start with a digit (board size like "37" or "37 Blitz")
                if line[0].isdigit():
                    return "notation"

        # Default to transcript if we can't determine
        return "transcript"

    def _get_delegate(self) -> ReplayLoader:
        """Get or create the appropriate loader delegate.

        Returns:
            The loader instance for the detected format
        """
        if self._delegate is None:
            file_format = self.detect_file_format(self.filename)

            if file_format == "sgf":
                self._delegate = SGFLoader(self.filename, self._status_reporter)
            elif file_format == "notation":
                self._delegate = NotationLoader(self.filename, self._status_reporter)
            else:  # transcript
                self._delegate = TranscriptLoader(self.filename, self._status_reporter)

        return self._delegate

    # ReplayLoader protocol implementation ----------------------------------------

    @property
    def detected_rings(self) -> int:
        """Board size detected from replay file."""
        return self._get_delegate().detected_rings

    @property
    def blitz(self) -> bool:
        """Whether this is a Blitz variant game."""
        return self._get_delegate().blitz

    @property
    def player1_name(self) -> str | None:
        """Player 1 name if available."""
        return self._get_delegate().player1_name

    @property
    def player2_name(self) -> str | None:
        """Player 2 name if available."""
        return self._get_delegate().player2_name

    def load(self) -> tuple[list[dict], list[dict]]:
        """Load and parse the replay file.

        Returns:
            Tuple of (player1_actions, player2_actions) where each action is a dict
        """
        return self._get_delegate().load()

    def set_status_reporter(self, reporter: Callable[[str], None] | None) -> None:
        """Set or update the status reporter callback."""
        self._status_reporter = reporter
        if self._delegate is not None:
            self._delegate.set_status_reporter(reporter)