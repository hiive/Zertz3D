"""Game controller for Zertz 3D.

Manages game loop, player actions, renderer updates, and game state.
"""
from __future__ import annotations

from game.zertz_game import PLAYER_1_WIN, PLAYER_2_WIN
from controller.replay_loader import ReplayLoader
from controller.game_logger import GameLogger
from controller.action_text_formatter import ActionTextFormatter
from controller.game_session import GameSession
from controller.game_loop import GameLoop
from shared.interfaces import IRenderer, IRendererFactory

class ZertzGameController:

    def __init__(self, rings=37, replay_file=None, seed=None, log_to_file=False, partial_replay=False,
                 max_games=None, show_moves=False, show_coords=False, log_notation=False,
                 blitz=False, move_duration=0.666,
                 renderer_or_factory: IRenderer | IRendererFactory | None = None):
        self.show_coords = show_coords
        self.max_games = max_games  # None means play indefinitely
        self.show_moves = show_moves
        self.move_duration = move_duration

        # Renderer state tracking
        self.renderer = None
        self.waiting_for_renderer = False
        self._completion_queue = []
        self._game_ending_processed = False  # Track if current game ending has been processed

        # Initialize logger and move formatter
        self.logger = GameLogger(log_to_file=log_to_file, log_notation=log_notation, status_reporter=self._report)
        self.move_formatter = ActionTextFormatter()

        # Load replay first to detect board size and variant if needed
        replay_actions = None
        loader = None
        if replay_file is not None:
            loader = ReplayLoader(replay_file, blitz=blitz, status_reporter=self._report)
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
            status_reporter=self._report
        )


        if isinstance(renderer_or_factory, IRenderer):
            self.renderer = renderer_or_factory
        elif isinstance(renderer_or_factory, IRendererFactory):
            self.renderer = renderer_or_factory(self)
        else:
            raise TypeError("renderer_or_factory must be an IRenderer, IRendererFactory, or None")

        self._game_loop = GameLoop(self, self.renderer, self.move_duration)

        # Ensure dependent components use shared status reporter
        self.logger.set_status_reporter(self._report)
        self.session.set_status_reporter(self._report)
        if loader is not None:
            loader.set_status_reporter(self._report)

        # Open log file for first game (if not in replay mode)
        if not self.session.is_replay_mode():
            self.logger.open_log_files(self.session.get_seed(), self.session.rings, self.session.blitz)

    def _close_log_file(self):
        """Close the current log file and append final game state."""
        if self.logger.log_file is not None or self.logger.notation_file is not None:
            self.logger.close_log_files(self.session.game)

    def _log_action(self, player_num, action_dict):
        """Log an action to the file if logging is enabled."""
        self.logger.log_action(player_num, action_dict)

    def _log_notation(self, notation):
        """Log a move in official notation to the notation file."""
        self.logger.log_notation(notation)

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

        # Open log file for new game (if not in replay mode)
        if not self.session.is_replay_mode():
            self.logger.open_log_files(self.session.get_seed(), self.session.rings, self.session.blitz)

    def _display_valid_moves(self, player):
        """Display valid move information for the current player.

        Args:
            player: The current player
        """
        formatted_moves = self.move_formatter.format_valid_actions(self.session.game, player)
        self._report(formatted_moves)

    def update_game(self, task):
        # Wait for renderer callback before continuing
        if self.waiting_for_renderer:
            return task.again

        # Process completed renderer actions (if any) before taking new actions
        self._process_completion_queue(task.delay_time)

        # Always check game status after processing completions (game may have ended)
        status = self._check_game_status(task)
        if status == task.done:
            return task.done

        # Get next action from current player
        player = self.session.get_current_player()

        try:
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

        # Display valid moves if enabled
        if self.show_moves:
            self._display_valid_moves(player)

        # Convert action to string and log
        try:
            _, action_dict = self.session.game.action_to_str(ax, ay)
        except IndexError as e:
            self._report(f"Error converting action to string: {e}")
            self._report(f"Action type: {ax}, Action: {ay}")
            raise

        # Log action to normal log
        self._log_action(player.n, action_dict)

        # Get render data BEFORE executing action (so board state is pre-action)
        # This encapsulates all data transformations in the game layer (Recommendation 1)
        render_data = self.session.game.get_render_data(ax, ay, self.show_moves)

        # Execute action FIRST to get action_result
        action_result = self.session.game.take_action(ax, ay)

        # Generate complete notation WITH action_result (includes isolation in one pass)
        notation = self.session.game.action_to_notation(action_dict, action_result)
        self._report(f'Player {player.n}: {action_dict} ({notation})')

        # Log notation directly (no buffering needed)
        self._log_notation(notation)

        # Dispatch to renderer (if present) and wait for callback
        if self.renderer:
            self.waiting_for_renderer = True
            self.renderer.execute_action(player, render_data, action_result, task.delay_time, self._handle_action_completion)
            if self.waiting_for_renderer:
                return task.again
            # If callback was synchronous, process completions and check status immediately
            self._process_completion_queue(task.delay_time)
            return self._check_game_status(task)

        # Headless mode: process action completion immediately
        self._handle_action_completion(player, action_result)
        self._process_completion_queue(task.delay_time)
        return self._check_game_status(task)

    def _process_action_result(self, player, result, delay_time):
        """Apply post-animation effects for a completed action."""
        if result.is_isolation():
            for removal in result.captured_marbles:
                if removal['marble'] is not None:
                    player.add_capture(removal['marble'])
                if self.renderer is not None:
                    self.renderer.show_isolated_removal(player, removal['pos'], removal['marble'], delay_time)
        elif result.has_captures():
            player.add_capture(result.captured_marbles)

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
            self._process_action_result(player, result, delay_time)
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

        self._report("")
        if winner:
            # Convert winner constant to player number for display
            player_num = 1 if winner == PLAYER_1_WIN else 2
            self._report(f'Winner: Player {player_num}')
        else:
            self._report('Game ended in a tie')
        self.session.game.print_state(self._report)

        self._report(f"Player 1 captures: {self.session.player1.captured}")
        self._report(f"Player 2 captures: {self.session.player2.captured}")

        # Increment games played counter
        self.session.increment_games_played()

        if self.session.is_replay_mode():
            self._report("Replay complete")
            return task.done
        if self.max_games is not None and self.session.get_games_played() >= self.max_games:
            # Reached max games limit
            self._report(f"Completed {self.session.get_games_played()} game(s)")
            self._close_log_file()  # Close log file before exiting
            return task.done

        # Continue with next game
        self._reset_board()
        return task.again

    def _report(self, message: str | None) -> None:
        """Forward status messages to the active renderer(s) or stdout."""
        if message is None:
            return
        text = str(message)
        if self.renderer is not None:
            self.renderer.report_status(text)
        else:
            print(text)
