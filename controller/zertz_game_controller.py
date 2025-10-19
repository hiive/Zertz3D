"""Game controller for Zertz 3D.

Manages game loop, player actions, renderer updates, and game state.
"""

from __future__ import annotations

import time
import numpy as np

from game.zertz_game import PLAYER_1_WIN, PLAYER_2_WIN
from game.loaders import TranscriptLoader, NotationLoader
from controller.game_logger import GameLogger
from controller.action_text_formatter import ActionTextFormatter
from controller.action_processor import ActionProcessor
from controller.game_session import GameSession
from controller.game_loop import GameLoop
from controller.human_player_interaction_manager import HumanPlayerInteractionManager
from shared.interfaces import IRenderer, IRendererFactory


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
        track_statistics=False,
        mcts_player2_iterations: int | None = None,
    ):
        self.show_coords = show_coords
        self.max_games = max_games  # None means play indefinitely
        self.highlight_choices = highlight_choices
        self.move_duration = move_duration
        self.animation_duration = move_duration * self.ANIMATION_DURATION_RATIO

        # Statistics tracking
        self.track_statistics = track_statistics
        self.game_stats = []  # List of game durations in seconds
        self.win_loss_stats = {PLAYER_1_WIN: 0, PLAYER_2_WIN: 0, 0: 0}  # Win/loss/tie counts
        self.current_game_start_time = None
        self.total_start_time = None

        # Renderer state tracking
        self.renderer = None
        self.waiting_for_renderer = False
        self._completion_queue = []
        self._game_ending_processed = (
            False  # Track if current game ending has been processed
        )

        self.move_formatter = ActionTextFormatter()

        # Load replay first to detect board size and variant if needed
        replay_actions = None
        loader = None
        if replay_file is not None:
            # Auto-detect file format (transcript or notation)
            file_format = self._detect_file_format(replay_file)
            if file_format == "notation":
                loader = NotationLoader(replay_file, status_reporter=print)
            else:  # transcript format
                loader = TranscriptLoader(replay_file, status_reporter=print)

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
            status_reporter=print,
            human_players=human_players,
            mcts_player2_iterations=mcts_player2_iterations,
        )

        # Create logger with all configuration - it manages all writers internally
        self.logger = GameLogger(
            session=self.session,
            transcript_dir=log_to_file,
            notation_dir=log_notation_to_file,
            log_to_screen=log_to_screen,
            log_notation_to_screen=log_notation_to_screen,
            status_reporter=self._report,
        )

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

        # Create interaction manager for handling human player feedback
        self.interaction_manager = HumanPlayerInteractionManager(
            self.renderer, self.session
        )

        if self.renderer and hasattr(self.renderer, "set_selection_callback"):
            self.renderer.set_selection_callback(
                self.interaction_manager.handle_renderer_selection
            )
        if self.renderer and hasattr(self.renderer, "set_hover_callback"):
            self.renderer.set_hover_callback(
                self.interaction_manager.handle_renderer_hover
            )

        self.action_processor = ActionProcessor(self.renderer)
        self._game_loop = GameLoop(self, self.renderer, self.move_duration)

        # Ensure dependent components use shared status reporter
        self.logger.set_status_reporter(self._report)
        self.session.set_status_reporter(self._report)
        if loader is not None:
            loader.set_status_reporter(self._report)

        # Report log filenames created during initialization
        for filename in self.logger.get_log_filenames():
            self._report(f"Logging to: {filename}")

        # Start logging for first game (logger handles all internal checks)
        self.logger.start_log(
            self.session.get_seed(), self.session.rings, self.session.blitz
        )

        # Start timing for first game if enabled
        if self.track_statistics:
            self.total_start_time = time.time()
            self.current_game_start_time = time.time()

    def _close_log_file(self):
        """Close the current log file and append final game state."""
        self.logger.end_log(self.session.game)

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

        # Reset renderer if present
        if self.renderer is not None:
            self.renderer.reset_board()

        # Start logging for new game - logger handles writer recreation internally
        self.logger.start_log(
            self.session.get_seed(), self.session.rings, self.session.blitz
        )

        # Start timing for new game if enabled
        if self.track_statistics:
            self.current_game_start_time = time.time()

    def _display_valid_moves(self, player):
        """Display valid move information for the current player.

        Args:
            player: The current player
        """
        formatted_moves = self.move_formatter.format_valid_actions(
            self.session.game, player
        )
        self._report(formatted_moves)

    def _resolve_player_context(self, player, placement_mask, capture_mask):
        # Determine the logical phase name based on available masks
        if np.any(capture_mask):
            context = "capture"
        elif np.any(placement_mask):
            context = "placement"
        else:
            context = "idle"
        player.on_turn_start(context, placement_mask, capture_mask)
        self.interaction_manager.update_hover_feedback(player)

    def _update_context_highlights(self, player, placement_mask, capture_mask):
        if not self.highlight_choices or not self.renderer:
            return

        self.renderer.clear_highlight_context()
        self.renderer.apply_context_masks(
            self.session.game.board, placement_mask, capture_mask
        )
        self.interaction_manager.update_hover_feedback(player)

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
                if self.session.is_partial_replay():
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
        self.interaction_manager.clear_hover_feedback()

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

        # TEMPORARY: Call canonicalize_state for profiling - REMOVE AFTER PROFILING
        self.session.game.board.canonicalize_state()
        # END TEMPORARY

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

        # Record game time and outcome if statistics tracking is enabled
        if self.track_statistics and self.current_game_start_time is not None:
            game_duration = time.time() - self.current_game_start_time
            self.game_stats.append(game_duration)
            # Record win/loss/tie
            self.win_loss_stats[game_over] += 1

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

    def print_statistics(self) -> None:
        """Print timing and win/loss statistics for all games played.

        Outputs mean, min, max, standard deviation for individual games,
        plus total execution time and win/loss/tie breakdown.
        """
        if not self.track_statistics or not self.game_stats:
            return

        import statistics

        # Calculate total execution time
        total_time = time.time() - self.total_start_time if self.total_start_time else 0

        # Calculate timing statistics
        mean_time = statistics.mean(self.game_stats)
        min_time = min(self.game_stats)
        max_time = max(self.game_stats)
        std_time = statistics.stdev(self.game_stats) if len(self.game_stats) > 1 else 0.0

        # Get win/loss stats
        total_games = len(self.game_stats)
        player1_wins = self.win_loss_stats[PLAYER_1_WIN]
        player2_wins = self.win_loss_stats[PLAYER_2_WIN]
        ties = self.win_loss_stats[0]

        # Calculate percentages
        p1_pct = (player1_wins / total_games * 100) if total_games > 0 else 0
        p2_pct = (player2_wins / total_games * 100) if total_games > 0 else 0
        tie_pct = (ties / total_games * 100) if total_games > 0 else 0

        # Print statistics
        print("\n" + "=" * 60)
        print("STATISTICS")
        print("=" * 60)
        print(f"Games played: {total_games}")
        print()
        print("Win/Loss/Tie:")
        print(f"  Player 1 wins: {player1_wins} ({p1_pct:.1f}%)")
        print(f"  Player 2 wins: {player2_wins} ({p2_pct:.1f}%)")
        print(f"  Ties: {ties} ({tie_pct:.1f}%)")
        print()
        print("Timing:")
        print(f"  Mean time per game: {mean_time:.3f}s")
        print(f"  Min time: {min_time:.3f}s")
        print(f"  Max time: {max_time:.3f}s")
        print(f"  Std deviation: {std_time:.3f}s")
        print(f"  Total execution time: {total_time:.3f}s")
        print("=" * 60)
