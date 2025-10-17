"""Game controller for Zertz 3D.

Manages game loop, player actions, renderer updates, and game state.
"""

from __future__ import annotations

from typing import Optional
import sys
import os

import numpy as np

from game.zertz_game import PLAYER_1_WIN, PLAYER_2_WIN
from game.loaders import TranscriptLoader, NotationLoader
from game.writers import NotationWriter, TranscriptWriter
from controller.game_logger import GameLogger
from controller.action_text_formatter import ActionTextFormatter
from controller.action_processor import ActionProcessor
from controller.game_session import GameSession
from controller.game_loop import GameLoop
from shared.interfaces import IRenderer, IRendererFactory
from shared.constants import MARBLE_TYPES


class ZertzGameController:
    # Fraction of move_duration used for animations (leaves buffer before next turn)
    ANIMATION_DURATION_RATIO = 60.0 / 100.0  # 1% of 60fps

    @staticmethod
    def _detect_file_format(filepath: str) -> str:
        """Detect whether a file is transcript or notation format.

        Transcript files start with '# Seed' or 'Player'.
        Notation files start with a digit (board size).

        Args:
            filepath: Path to the file to detect

        Returns:
            "transcript" or "notation"
        """
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

    def __init__(
        self,
        rings=37,
        replay_file=None,
        seed=None,
        log_to_file: str | None = None,
        log_to_screen=False,
        log_notation_to_file: str | None = None,
        log_notation_to_screen=False,
        partial_replay=False,
        max_games=None,
        highlight_choices=False,
        show_coords=False,
        blitz=False,
        move_duration=0.666,
        renderer_or_factory: IRenderer | IRendererFactory | None = None,
        human_players: tuple[int, ...] | None = None,
    ):
        self.show_coords = show_coords
        self.max_games = max_games  # None means play indefinitely
        self.highlight_choices = highlight_choices
        self.move_duration = move_duration
        self.animation_duration = move_duration * self.ANIMATION_DURATION_RATIO

        # Renderer state tracking
        self.renderer = None
        self.waiting_for_renderer = False
        self._completion_queue = []
        self._game_ending_processed = (
            False  # Track if current game ending has been processed
        )

        self.move_formatter = ActionTextFormatter()

        # Create logger early (before session) so _report() can use it during initialization
        # Start with empty writers list - we'll add them after session is created
        self.logger = GameLogger(writers=[], status_reporter=self._report)
        self._log_filenames = []  # Track filenames for status reporting
        self._log_to_file = log_to_file
        self._log_to_screen = log_to_screen
        self._log_notation_to_file = log_notation_to_file
        self._log_notation_to_screen = log_notation_to_screen

        # Load replay first to detect board size and variant if needed
        replay_actions = None
        loader = None
        if replay_file is not None:
            # Auto-detect file format (transcript or notation)
            file_format = self._detect_file_format(replay_file)
            if file_format == "notation":
                loader = NotationLoader(replay_file, status_reporter=self._report)
            else:  # transcript format
                loader = TranscriptLoader(replay_file, status_reporter=self._report)

            replay_actions = loader.load()
            # Use loader's authoritative configuration
            rings = loader.detected_rings
            blitz = loader.blitz

        # Create game session (handles game instance, players, seed)
        self.session = GameSession(
            rings=rings,
            blitz=blitz,
            seed=seed,
            replay_actions=replay_actions,
            partial_replay=partial_replay,
            t=5,
            status_reporter=self._report,
            human_players=human_players,
        )

        # Now that session exists, configure logger with appropriate writers
        writers = []

        # File writers (only for non-replay mode)
        if self._log_to_file and not self.session.is_replay_mode():
            # Create directory if it doesn't exist
            os.makedirs(self._log_to_file, exist_ok=True)
            variant = "_blitz" if self.session.blitz else ""
            filename = f"zertzlog{variant}_{self.session.get_seed()}.txt"
            filepath = os.path.join(self._log_to_file, filename)
            self._log_filenames.append(filepath)
            writers.append(TranscriptWriter(open(filepath, "w")))
        if self._log_notation_to_file and not self.session.is_replay_mode():
            # Create directory if it doesn't exist
            os.makedirs(self._log_notation_to_file, exist_ok=True)
            variant = "_blitz" if self.session.blitz else ""
            filename = f"zertzlog{variant}_{self.session.get_seed()}_notation.txt"
            filepath = os.path.join(self._log_notation_to_file, filename)
            self._log_filenames.append(filepath)
            writers.append(NotationWriter(open(filepath, "w")))

        # Screen writers (for all modes including replay)
        if self._log_to_screen:
            writers.append(TranscriptWriter(sys.stdout))
        if self._log_notation_to_screen:
            writers.append(NotationWriter(sys.stdout))

        self.logger.writers = writers

        if isinstance(renderer_or_factory, IRenderer):
            self.renderer = renderer_or_factory
        elif isinstance(renderer_or_factory, IRendererFactory):
            self.renderer = renderer_or_factory(self)
        elif renderer_or_factory is None:
            pass
        else:
            raise TypeError(
                "renderer_or_factory must be an IRenderer, IRendererFactory, or None"
            )

        if self.renderer and hasattr(self.renderer, "set_selection_callback"):
            self.renderer.set_selection_callback(self._handle_renderer_selection)
        if self.renderer and hasattr(self.renderer, "set_hover_callback"):
            self.renderer.set_hover_callback(self._handle_renderer_hover)

        self.action_processor = ActionProcessor(self.renderer)
        self._game_loop = GameLoop(self, self.renderer, self.move_duration)

        # Ensure dependent components use shared status reporter
        self.logger.set_status_reporter(self._report)
        self.session.set_status_reporter(self._report)
        if loader is not None:
            loader.set_status_reporter(self._report)

        # Report log filenames for first game
        for filename in self._log_filenames:
            self._report(f"Logging to: {filename}")

        # Start logging for first game (screen writers work in all modes)
        if self.logger.writers:
            self.logger.start_game(
                self.session.get_seed(), self.session.rings, self.session.blitz
            )

    def _close_log_file(self):
        """Close the current log file and append final game state."""
        if self.logger.writers:
            self.logger.end_game(self.session.game)

    def _log_action(self, player_num, action_dict, action_result=None):
        """Log an action to all writers.

        Args:
            player_num: Player number (1 or 2)
            action_dict: Dictionary containing action details
            action_result: Optional ActionResult for notation formatting
        """
        self.logger.log_action(player_num, action_dict, action_result)

    def run(self):
        # self._game_loop = GameLoop(self, self.renderer, self.move_duration)
        self._game_loop.run()

    def _reset_board(self):
        """Reset the board for a new game."""
        # Close previous log file if it exists
        self._close_log_file()

        # Reset game ending flag for new game
        self._game_ending_processed = False

        # Reset game session (creates new game instance and players)
        self.session.reset_game()

        # Update writers for new game
        writers = []
        self._log_filenames = []
        if self.logger.writers:
            # Recreate writers - file writers get new files, screen writers persist
            for writer in self.logger.writers:
                # Screen writers (stdout) persist across games
                if writer.output == sys.stdout:
                    writers.append(writer)
                    continue

                # File writers get new files for each game
                variant = "_blitz" if self.session.blitz else ""
                if isinstance(writer, TranscriptWriter):
                    filename = f"zertzlog{variant}_{self.session.get_seed()}.txt"
                    filepath = os.path.join(self._log_to_file, filename)
                    self._log_filenames.append(filepath)
                    writers.append(TranscriptWriter(open(filepath, "w")))
                    self._report(f"Logging to: {filepath}")
                elif isinstance(writer, NotationWriter):
                    filename = f"zertzlog{variant}_{self.session.get_seed()}_notation.txt"
                    filepath = os.path.join(self._log_notation_to_file, filename)
                    self._log_filenames.append(filepath)
                    writers.append(NotationWriter(open(filepath, "w")))
                    self._report(f"Logging notation to: {filepath}")
            self.logger.writers = writers

        # Reset renderer if present
        if self.renderer is not None:
            self.renderer.reset_board()

        # Start logging for new game (screen writers work in all modes)
        if self.logger.writers:
            self.logger.start_game(
                self.session.get_seed(), self.session.rings, self.session.blitz
            )

    def _display_valid_moves(self, player):
        """Display valid move information for the current player.

        Args:
            player: The current player
        """
        formatted_moves = self.move_formatter.format_valid_actions(
            self.session.game, player
        )
        self._report(formatted_moves)

    def _clear_hover_feedback(self):
        if self.renderer and hasattr(self.renderer, "clear_hover_highlights"):
            self.renderer.clear_hover_highlights()

    def _update_hover_feedback(self, player) -> None:
        if not self.renderer or not hasattr(self.renderer, "show_hover_feedback"):
            return
        if not hasattr(player, "get_current_options") or not hasattr(
            player, "get_selection_state"
        ):
            self._clear_hover_feedback()
            return

        options = player.get_current_options()
        if not options:
            self._clear_hover_feedback()
            return

        board = self.session.game.board
        state = player.get_selection_state()
        primary: set[str] = set()
        secondary: set[str] = set()
        supply_colors: set[str] = set()
        captured_targets: set[tuple[int, str]] = set()
        player_id = getattr(player, "n", None)

        def idx_to_label(idx: tuple[int, int] | None) -> str | None:
            if idx is None:
                return None
            y, x = idx
            try:
                label = board.index_to_str((y, x))
            except (IndexError, ValueError):
                return None
            return label or None

        placement_mask, capture_mask = (
            player.get_context_masks()
            if hasattr(player, "get_context_masks")
            else (None, None)
        )
        context = options.get("context")

        if context == "placement":
            color = state.get("placement_color")
            color_idx = state.get("placement_color_idx")
            if color_idx is None:
                # Highlight all available supply colors
                counts = options.get("supply_counts", ())
                for idx, marble in enumerate(MARBLE_TYPES):
                    if idx < len(counts) and counts[idx] > 0:
                        supply_colors.add(marble)
            else:
                # Check if a specific marble is selected (has supply_key or captured_key)
                supply_key = state.get("placement_supply_key")
                captured_key = state.get("placement_captured_key")

                if supply_key:
                    # Highlight the specific supply marble, not all marbles of that color
                    primary.add(supply_key)
                elif captured_key:
                    # Highlight the specific captured marble, not all marbles of that color
                    primary.add(captured_key)
                else:
                    # Fallback: highlight all marbles of the selected color
                    if color:
                        supply_colors.add(color)
                    if (
                        player_id is not None
                        and state.get("placement_source") == "captured"
                        and color
                    ):
                        captured_targets.add((player_id, color))
            if player_id is not None and options.get("supply_total", 0) == 0:
                for marble in options.get("supply_colors", set()):
                    captured_targets.add((player_id, marble))
            placement_dst = state.get("placement_dst")
            if placement_dst is not None:
                label = idx_to_label(placement_dst)
                if label:
                    primary.add(label)
            pending_removals = state.get("placement_pending_removals") or set()
            for idx in pending_removals:
                label = idx_to_label(idx)
                if label:
                    secondary.add(label)

            if (
                color_idx is not None
                and placement_mask is not None
                and placement_mask.size > 0
            ):
                width = board.width
                if placement_dst is None:
                    dest_indices = np.where(np.any(placement_mask[color_idx], axis=1))[
                        0
                    ]
                    for flat in dest_indices:
                        y, x = divmod(int(flat), width)
                        label = idx_to_label((y, x))
                        if label:
                            primary.add(label)
                elif not pending_removals:
                    allow_none = state.get("placement_allow_none_removal", False)
                    if allow_none:
                        label = idx_to_label(placement_dst)
                        if label:
                            secondary.add(label)

        elif context == "capture":
            capture_src = state.get("capture_src")
            if capture_src is not None:
                label = idx_to_label(capture_src)
                if label:
                    primary.add(label)
                capture_paths = options.get("capture_paths", {}).get(capture_src, set())
                for idx in capture_paths:
                    label_dst = idx_to_label(idx)
                    if label_dst:
                        secondary.add(label_dst)
            else:
                for idx in options.get("capture_sources", set()):
                    label = idx_to_label(idx)
                    if label:
                        primary.add(label)

        hover_state = state.get("hover")
        if hover_state:
            h_type = hover_state.get("type")
            if h_type in ("ring", "board_marble"):
                label = idx_to_label(hover_state.get("index"))
                if label:
                    if context == "capture" and state.get("capture_src") is not None:
                        secondary.add(label)
                    elif (
                        context == "placement"
                        and state.get("placement_dst") is not None
                    ):
                        secondary.add(label)
                    else:
                        primary.add(label)
            # Note: Supply/captured marble hovering is now handled above by checking
            # for specific marble keys (supply_key/captured_key) and adding them to
            # primary highlights instead of supply_colors/captured_targets, which
            # would highlight ALL marbles of that color instead of just the selected one.

        if not primary and not secondary and not supply_colors and not captured_targets:
            self._clear_hover_feedback()
            return

        self.renderer.show_hover_feedback(
            primary, secondary, supply_colors, captured_targets
        )

    def _handle_renderer_selection(self, selection: dict) -> None:
        player = self.session.get_current_player()
        payload = dict(selection)
        label = selection.get("label")
        if label:
            try:
                payload["index"] = self.session.game.board.str_to_index(label)
            except ValueError:
                payload["index"] = None

        index = payload.get("index")
        if index is not None:
            payload["index"] = (int(index[0]), int(index[1]))

        player.handle_selection(payload)
        self._update_hover_feedback(player)

    def _handle_renderer_hover(self, hover: Optional[dict]) -> None:
        player = self.session.get_current_player()
        if not hasattr(player, "handle_hover"):
            return

        payload = None
        if hover:
            payload = dict(hover)
            label = hover.get("label")
            if label:
                try:
                    payload["index"] = self.session.game.board.str_to_index(label)
                except ValueError:
                    payload["index"] = None
            index = payload.get("index")
            if index is not None:
                payload["index"] = (int(index[0]), int(index[1]))

        player.handle_hover(payload)
        self._update_hover_feedback(player)

    def _resolve_player_context(self, player, placement_mask, capture_mask):
        # Determine the logical phase name based on available masks
        if np.any(capture_mask):
            context = "capture"
        elif np.any(placement_mask):
            context = "placement"
        else:
            context = "idle"
        player.on_turn_start(context, placement_mask, capture_mask)
        self._update_hover_feedback(player)

    def _update_context_highlights(self, player, placement_mask, capture_mask):
        if not self.highlight_choices or not self.renderer:
            return

        self.renderer.clear_highlight_context()
        self.renderer.apply_context_masks(
            self.session.game.board, placement_mask, capture_mask
        )
        self._update_hover_feedback(player)

    def update_game(self, task):
        # Wait for renderer callback before continuing
        if self.waiting_for_renderer:
            return task.again

        # Process completed renderer actions (if any) before taking new actions
        self._process_completion_queue(self.animation_duration)

        # Always check game status after processing completions (game may have ended)
        status = self._check_game_status(task)
        if status == task.done:
            return task.done

        # Get next action from current player
        player = self.session.get_current_player()

        placement_mask, capture_mask = self.session.game.get_valid_actions()
        self._resolve_player_context(player, placement_mask, capture_mask)

        if self.highlight_choices and self.renderer:
            self._update_context_highlights(player, placement_mask, capture_mask)

        try:
            if (
                hasattr(player, "pending_actions_empty")
                and player.pending_actions_empty()
            ):
                return task.again
            ax, ay = player.get_action()
        except ValueError as e:
            self._report(f"Error getting action: {e}")
            if self.session.is_replay_mode():
                if self.session.partial_replay:
                    # Switch to random play
                    player = self.session.switch_to_random_play(player)
                    ax, ay = player.get_action()
                else:
                    self._report("Replay finished")
                    return task.done
            else:
                raise

        # Clear context highlights once an action is chosen
        if self.highlight_choices and self.renderer:
            self.renderer.clear_highlight_context()
        player.clear_context()
        self._clear_hover_feedback()

        # Display valid moves if enabled (text-only)
        if self.highlight_choices:
            self._display_valid_moves(player)

        # Convert action to string
        try:
            _, action_dict = self.session.game.action_to_str(ax, ay)
        except IndexError as e:
            self._report(f"Error converting action to string: {e}")
            self._report(f"Action type: {ax}, Action: {ay}")
            raise

        # Get render data BEFORE executing action (so board state is pre-action)
        # This encapsulates all data transformations in the game layer (Recommendation 1)
        render_data = self.session.game.get_render_data(ax, ay, self.highlight_choices)

        # Execute action FIRST to get action_result
        action_result = self.session.game.take_action(ax, ay)

        # Generate complete notation WITH action_result (includes isolation in one pass)
        # notation = self.session.game.action_to_notation(action_dict, action_result)
        # self._report(f"Player {player.n}: {action_dict} ({notation})")
        # 
        # Log action with action_result (used by NotationWriter for isolation captures)
        self._log_action(player.n, action_dict, action_result)

        # Dispatch to renderer (if present) and wait for callback
        if self.renderer:
            self.waiting_for_renderer = True
            self.renderer.execute_action(
                player,
                render_data,
                action_result,
                self.animation_duration,
                self._handle_action_completion,
            )
            if self.waiting_for_renderer:
                return task.again
            # If callback was synchronous, process completions and check status immediately
            self._process_completion_queue(self.animation_duration)
            return self._check_game_status(task)

        # Headless mode: process action completion immediately
        self._handle_action_completion(player, action_result)
        self._process_completion_queue(self.animation_duration)
        return self._check_game_status(task)

    def _handle_action_completion(self, player, action_result):
        """Renderer callback invoked when all visuals for an action are complete."""
        self.waiting_for_renderer = False
        self._completion_queue.append((player, action_result))

    def _process_completion_queue(self, delay_time):
        """Apply all pending action results.

        Returns:
            bool: True if at least one completion was processed.
        """
        processed = False
        while self._completion_queue:
            player, result = self._completion_queue.pop(0)
            self.action_processor.process(player, result, delay_time)
            processed = True
        return processed

    def _check_game_status(self, task):
        """Check if game is over and handle the ending if needed.

        Returns:
            task.done if game should stop, task.again if game should continue
        """
        game_over = self.session.game.get_game_ended()
        if game_over is None:
            return task.again

        # Game is over - handle the ending (prints winner, increments counter, etc.)
        return self._handle_game_ending(game_over, task)

    def _handle_game_ending(self, game_over, task):
        """Handle game ending with side effects (print winner, increment counter, etc.).

        This method is idempotent - calling it multiple times for the same game
        has no additional effect (prevents duplicate output in graphical mode).

        Args:
            game_over: Game result (PLAYER_1_WIN, PLAYER_2_WIN, or TIE)
            task: Task object for returning status

        Returns:
            task.done if should exit, task.again if should continue with next game
        """
        # If we've already handled this game's ending, just return done
        # (prevents duplicate output when callbacks cause multiple calls)
        if self._game_ending_processed:
            return task.done

        # Mark that we're handling this game's ending
        self._game_ending_processed = True

        winner = game_over if game_over in [PLAYER_1_WIN, PLAYER_2_WIN] else None

        # Get detailed reason for game ending
        reason = self.session.game.get_game_end_reason()

        self._report("")
        if winner:
            # Convert winner constant to player number for display
            player_num = 1 if winner == PLAYER_1_WIN else 2
            self._report(f"Winner: Player {player_num} ({reason})")
        else:
            self._report(f"Game ended in a tie ({reason})")
        # Note: Board state is written by TranscriptWriter.write_footer()
        # to avoid duplication

        self._report(f"Player 1 captures: {self.session.player1.captured}")
        self._report(f"Player 2 captures: {self.session.player2.captured}")

        # Increment games played counter
        self.session.increment_games_played()

        if self.session.is_replay_mode():
            self._report("Replay complete")
            return task.done
        if (
            self.max_games is not None
            and self.session.get_games_played() >= self.max_games
        ):
            # Reached max games limit
            self._report(f"Completed {self.session.get_games_played()} game(s)")
            self._close_log_file()  # Close log file before exiting
            return task.done

        # Continue with next game
        self._reset_board()
        return task.again

    def _report(self, message: str | None) -> None:
        """Forward status messages to logger (sole handler for all text output).

        Logger is the central hub for all text output:
        - TranscriptWriter (file): writes as comments to log files
        - TranscriptWriter (stdout): writes as comments to screen
        - NotationWriter: ignores comments (no-op)

        Note: Renderer is NOT sent status messages to avoid duplication when
        both transcript-screen and notation-screen are used together.
        TextRenderer only outputs "Executing action" via its execute_action method.
        """
        if message is None:
            return
        text = str(message)

        # Logger is the sole handler for all text output
        self.logger.log_comment(text)
