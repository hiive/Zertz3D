"""Game logging for Zertz 3D.

Handles logging game actions and notation to files.
"""

from typing import Callable


class GameLogger:
    """Handles logging to files (actions + notation)."""

    def __init__(self, log_to_file=False, log_notation=False, status_reporter: Callable[[str], None] | None = None):
        """Initialize the game logger.

        Args:
            log_to_file: Whether to log actions to file
            log_notation: Whether to log notation to file
        """
        self.log_to_file_enabled = log_to_file
        self.log_notation_enabled = log_notation
        self.log_file = None
        self.log_filename = None
        self.notation_file = None
        self.notation_filename = None
        self._status_reporter: Callable[[str], None] | None = status_reporter

    def open_log_files(self, seed, rings, blitz=False):
        """Open new log files for the current game.

        Args:
            seed: Random seed for this game
            rings: Number of rings on the board
            blitz: Whether this is a blitz game
        """
        variant = "_blitz" if blitz else ""

        if self.log_to_file_enabled:
            self.log_filename = f"zertzlog{variant}_{seed}.txt"
            self.log_file = open(self.log_filename, 'w')
            self.log_file.write(f"# Seed: {seed}\n")
            self.log_file.write(f"# Rings: {rings}\n")
            if blitz:
                self.log_file.write("# Variant: Blitz\n")
            self.log_file.write("#\n")
            self._report(f"Logging game to: {self.log_filename}")

        if self.log_notation_enabled:
            self.notation_filename = f"zertzlog{variant}_{seed}_notation.txt"
            self.notation_file = open(self.notation_filename, 'w')
            # First line: rings and variant
            variant_text = " Blitz" if blitz else ""
            self.notation_file.write(f"{rings}{variant_text}\n")
            self._report(f"Logging notation to: {self.notation_filename}")

    def close_log_files(self, game):
        """Close the current log files and append final game state.

        Args:
            game: ZertzGame instance to get final state from
        """
        if self.log_file is not None:
            # Append final game state as comments
            self.log_file.write("#\n")
            self.log_file.write("# Final game state:\n")
            self.log_file.write("# ---------------\n")
            self.log_file.write("# Board state:\n")
            board_state = game.board.state[0] + game.board.state[1] + game.board.state[2] * 2 + game.board.state[3] * 3
            for row in board_state:
                self.log_file.write(f"# {row}\n")
            self.log_file.write("# ---------------\n")
            self.log_file.write("# Marble supply:\n")
            self.log_file.write(f"# {game.board.state[-10:-1, 0, 0]}\n")
            self.log_file.write("# ---------------\n")
            self.log_file.close()
            self.log_file = None
            self._report(f"Game log saved to: {self.log_filename}")

        if self.notation_file is not None:
            self.notation_file.close()
            self.notation_file = None
            self._report(f"Notation log saved to: {self.notation_filename}")

    def log_action(self, player_num, action_dict):
        """Log an action to the file if logging is enabled.

        Args:
            player_num: Player number (1 or 2)
            action_dict: Dictionary containing action details
        """
        if self.log_file is not None:
            self.log_file.write(f"Player {player_num}: {action_dict}\n")
            self.log_file.flush()

    def log_notation(self, notation):
        """Log a move in official notation to the notation file.

        Args:
            notation: Notation string to log
        """
        if self.notation_file is not None:
            self.notation_file.write(f"{notation}\n")
            self.notation_file.flush()

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
