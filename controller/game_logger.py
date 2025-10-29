"""Game logging for Zertz 3D.

Handles logging game actions and notation using pluggable writers.
"""

import os
import sys
from typing import Callable

from game.writers import GameWriter, NotationWriter, TranscriptWriter


class GameLogger:
    """Manages multiple game action writers for flexible logging.

    Uses the Strategy pattern to support multiple output formats and destinations
    simultaneously (e.g., replay to file, notation to screen).

    Owns all writer lifecycle management including file creation, error handling,
    and writer recreation for new games.
    """

    def __init__(
        self,
        session,
        transcript_dir: str | None = None,
        notation_dir: str | None = None,
        log_to_screen: bool = False,
        log_notation_to_screen: bool = False,
        status_reporter: Callable[[str], None] | None = None,
    ):
        """Initialize the game logger.

        Args:
            session: GameSession instance (for blitz flag and replay mode checks)
            transcript_dir: Directory path for transcript log files (None to disable)
            notation_dir: Directory path for notation log files (None to disable)
            log_to_screen: Whether to log transcript to stdout
            log_notation_to_screen: Whether to log notation to stdout
            status_reporter: Optional callback for status messages
        """
        self.session = session
        self._transcript_dir = transcript_dir
        self._notation_dir = notation_dir
        self._log_to_screen = log_to_screen
        self._log_notation_to_screen = log_notation_to_screen
        self._status_reporter = status_reporter
        self._game_active = False

        # Track created log filenames
        self._log_filenames = []

        # Create initial writers
        self.writers: list[GameWriter] = []
        self._screen_writers: list[GameWriter] = []  # Keep references to screen writers
        self._create_initial_writers()

    def _create_file_writer(self, directory, filename, writer_class, log_type):
        """Create a file writer with robust error handling.

        Errors are reported to stderr. Controller should query get_log_filenames()
        after creation to report successful file creation to the user.

        Args:
            directory: Directory path to create file in
            filename: Name of file to create
            writer_class: Writer class to instantiate (TranscriptWriter or NotationWriter)
            log_type: Human-readable description for error messages (e.g., "transcript", "notation")

        Returns:
            Writer instance on success, None on failure
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(directory, exist_ok=True)
            filepath = os.path.join(directory, filename)
            self._log_filenames.append(filepath)
            return writer_class(open(filepath, "w"))
        except PermissionError as e:
            print(f"Error: Cannot create {log_type} log directory or file: {e}", file=sys.stderr)
            print(f"Attempted path: {directory}", file=sys.stderr)
            print(f"{log_type.capitalize()} logging to file disabled for this session", file=sys.stderr)
            return None
        except OSError as e:
            print(f"Error: Failed to create {log_type} log file: {e}", file=sys.stderr)
            print(f"Attempted path: {directory}", file=sys.stderr)
            print(f"{log_type.capitalize()} logging to file disabled for this session", file=sys.stderr)
            return None

    def _create_initial_writers(self):
        """Create initial set of writers based on configuration."""
        # Screen writers (persist across games)
        if self._log_to_screen:
            writer = TranscriptWriter(sys.stdout)
            self.writers.append(writer)
            self._screen_writers.append(writer)

        if self._log_notation_to_screen:
            writer = NotationWriter(sys.stdout)
            self.writers.append(writer)
            self._screen_writers.append(writer)

        # File writers (only for non-replay mode)
        if not self.session.is_replay_mode():
            self._create_game_file_writers()

    def _create_game_file_writers(self):
        """Create file writers for the current game."""
        variant = "_blitz" if self.session.blitz else ""

        if self._transcript_dir:
            filename = f"zertzlog{variant}_{self.session.get_seed()}.txt"
            writer = self._create_file_writer(
                self._transcript_dir, filename, TranscriptWriter, "transcript"
            )
            if writer:
                self.writers.append(writer)

        if self._notation_dir:
            filename = f"zertzlog{variant}_{self.session.get_seed()}_notation.txt"
            writer = self._create_file_writer(
                self._notation_dir, filename, NotationWriter, "notation"
            )
            if writer:
                self.writers.append(writer)

    def get_log_filenames(self):
        """Get list of log filenames created.

        Returns:
            List of absolute paths to created log files
        """
        return self._log_filenames.copy()

    def add_writer(self, writer: GameWriter) -> None:
        """Add a writer to the logger.

        Args:
            writer: GameWriter instance to add
        """
        self.writers.append(writer)

    def remove_writer(self, writer: GameWriter) -> None:
        """Remove a writer from the logger.

        Args:
            writer: GameWriter instance to remove
        """
        if writer in self.writers:
            self.writers.remove(writer)

    def start_log(self, seed: int, rings: int, blitz: bool = False) -> None:
        """Start logging for a new game and write headers to all writers.

        On first call, uses existing writers. On subsequent calls, recreates file writers
        while keeping screen writers. Does nothing if no writers are configured.

        Args:
            seed: Random seed for this game
            rings: Number of rings on the board
            blitz: Whether this is a blitz game
        """
        # Do nothing if no writers configured
        if not self.writers:
            return

        # If this isn't the first game, recreate file writers
        # (only when _game_active is True, meaning a game has already been logged)
        if self._game_active:
            # Close and remove file writers (but keep screen writers)
            new_writers = []
            for writer in self.writers:
                if writer in self._screen_writers:
                    new_writers.append(writer)
                else:
                    # Close file writer
                    writer.close()

            self.writers = new_writers

            # Create new file writers for the new game
            if not self.session.is_replay_mode():
                self._create_game_file_writers()

        # Write headers to all writers (include player names if available)
        player1_name = self.session.player1_config.name
        player2_name = self.session.player2_config.name

        for writer in self.writers:
            writer.write_header(seed, rings, blitz, player1_name, player2_name)

        self._game_active = True

    def end_log(self, game=None) -> None:
        """End logging for the current game, write footers and close file writers.

        Screen writers are NOT closed (they persist across games).
        File writers are removed from the writers list after closing.
        Does nothing if no writers are configured.

        Args:
            game: Optional ZertzGame instance for final state
        """
        if not self.writers:
            return

        # Write footers to all writers
        for writer in self.writers:
            writer.write_footer(game)

        # Close file writers and remove them from the list, keeping only screen writers
        new_writers = []
        for writer in self.writers:
            if writer in self._screen_writers:
                new_writers.append(writer)
            else:
                writer.close()
        self.writers = new_writers

        self._game_active = False

    def log_action(self, player_num: int, action_dict: dict, action_result=None) -> None:
        """Log an action to all writers.

        Args:
            player_num: Player number (1 or 2)
            action_dict: Dictionary containing action details
            action_result: Optional ActionResult (for isolation captures in notation)
        """
        for writer in self.writers:
            writer.write_action(player_num, action_dict, action_result)

    def log_comment(self, message: str) -> None:
        """Log a status/comment message to all writers.

        Args:
            message: Status message to log as a comment
        """
        for writer in self.writers:
            writer.write_comment(message)

    def set_status_reporter(self, reporter: Callable[[str], None] | None) -> None:
        """Set or update the status reporter callback."""
        self._status_reporter = reporter

    def _report(self, message: str | None) -> None:
        if message is None:
            return
        if self._status_reporter is not None:
            self._status_reporter(message)
        else:
            print(message)
