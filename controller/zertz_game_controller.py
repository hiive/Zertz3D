"""Game controller for Zertz 3D.

Manages game loop, player actions, renderer updates, and game state.
"""

from game.zertz_game import PLAYER_1_WIN, PLAYER_2_WIN
from renderer.zertz_renderer import ZertzRenderer
from controller.replay_loader import ReplayLoader
from controller.game_logger import GameLogger
from controller.action_text_renderer import ActionTextRenderer
from controller.game_session import GameSession
from game.zertz_board import ZertzBoard

class ZertzGameController:

    def __init__(self, rings=37, replay_file=None, seed=None, log_to_file=False, partial_replay=False, headless=False, max_games=None, show_moves=False, show_coords=False, log_notation=False, blitz=False, move_duration=0.666):
        self.show_coords = show_coords
        self.headless = headless
        self.max_games = max_games  # None means play indefinitely
        self.show_moves = show_moves
        self.move_duration = move_duration

        # Store action_dict for notation generation after action execution
        self.pending_action_dict = None
        self.pending_player = None
        self.pending_notation = None

        # Initialize logger and move formatter
        self.logger = GameLogger(log_to_file=log_to_file, log_notation=log_notation)
        self.move_formatter = ActionTextRenderer()

        # Load replay first to detect board size and variant if needed
        replay_actions = None
        if replay_file is not None:
            loader = ReplayLoader(replay_file, blitz=blitz)
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
            t=5
        )

        # Open log file for first game (if not in replay mode)
        if not self.session.is_replay_mode():
            self.logger.open_log_files(self.session.get_seed(), self.session.rings, self.session.blitz)

        # Create renderer with detected board size (only if not headless)
        board_layout = ZertzBoard.generate_standard_board_layout(self.session.rings)
        self.renderer = None if headless else ZertzRenderer(
            rings=self.session.rings,
            board_layout=board_layout,
            show_coords=self.show_coords,
            show_moves=show_moves,
            update_callback=self.update_game,
            move_duration=self.move_duration
        )

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
        if self.headless:
            self._run_headless()
        else:
            self.renderer.run()

    def _run_headless(self):
        """Run game loop without renderer."""
        # Create a simple task object for compatibility with update_game
        class SimpleTask:
            def __init__(self):
                self.delay_time = 0
                self.done = False
                self.again = True

        task = SimpleTask()

        # Run game until it's done
        while not task.done:
            result = self.update_game(task)
            if result == task.done:
                task.done = True

    def _reset_board(self):
        """Reset the board for a new game."""
        # Close previous log file if it exists
        self._close_log_file()

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
        print(formatted_moves)

    def update_game(self, task):
        # Check if renderer is still busy (animations or highlights)
        if self.renderer and self.renderer.is_busy():
            return task.again

        # Check if there's a completed action result from highlights
        result, player = None, None
        if self.renderer:
            result, player = self.renderer.get_pending_action_result()

        # If no pending result, get next action
        if result is None:
            # Get next action from current player
            player = self.session.get_current_player()

            try:
                ax, ay = player.get_action()
            except ValueError as e:
                print(f"Error getting action: {e}")
                if self.session.is_replay_mode():
                    if self.session.partial_replay:
                        # Switch to random play
                        player = self.session.switch_to_random_play(player)
                        ax, ay = player.get_action()
                    else:
                        print("Replay finished")
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
                print(f"Error converting action to string: {e}")
                print(f"Action type: {ax}, Action: {ay}")
                raise

            # Log action to normal log
            self._log_action(player.n, action_dict)

            # Generate notation (before executing action, so result is None for now)
            # Don't write to file yet - buffer it in case isolation updates it
            notation = self.session.game.action_to_notation(action_dict, None)
            print(f'Player {player.n}: {action_dict} ({notation})')

            # Buffer notation (logger will flush any previous notation before buffering new one)
            self.logger.buffer_notation(notation)
            self.pending_notation = notation  # Keep for console output

            # Store action_dict and player for later notation update if isolation occurs
            self.pending_action_dict = action_dict
            self.pending_player = player

            # Wait for renderer to finish any previous work
            if self.renderer and self.renderer.is_busy():
                return task.again

            # Get valid actions BEFORE executing (for highlighting)
            placement_before, capture_before = self.session.game.get_valid_actions()

            # Convert arrays to strings BEFORE executing action (so board state is pre-action)
            if self.show_moves:
                placement_positions = self.session.game.get_placement_positions(placement_before)
                capture_moves = self.session.game.get_capture_dicts(capture_before)
                removal_positions = self.session.game.get_removal_positions(placement_before, ax, ay)
            else:
                placement_positions = None
                capture_moves = None
                removal_positions = None

            # Execute action (game now provides frozen positions diff - Recommendation 1)
            action_result = self.session.game.take_action(ax, ay)

            # Pass action_result to renderer (no extraction needed - pass whole object)
            if self.renderer:
                self.renderer.execute_action(player, action_dict, action_result,
                                            placement_positions, capture_moves, removal_positions,
                                            task.delay_time)
                # Check if renderer started work (highlighting or animations)
                if self.renderer.is_busy():
                    return task.again

        # Process result (common path for both state machine and direct execution)
        else:
            # If action caused isolation, update the buffered notation
            if result.is_isolation() and self.pending_action_dict is not None:
                # Re-generate notation with ActionResult
                notation_with_isolation = self.session.game.action_to_notation(self.pending_action_dict, result)
                # Update the printed output (notation was already printed without isolation)
                print(f'  (Isolation occurred: {notation_with_isolation})')
                # Update the buffered notation in logger and controller
                self.logger.update_buffered_notation(notation_with_isolation)
                self.pending_notation = notation_with_isolation
                # Clear pending values
                self.pending_action_dict = None
                self.pending_player = None

            if result.is_isolation():
                for removal in result.captured_marbles:
                    if removal['marble'] is not None:
                        player.add_capture(removal['marble'])
                    if self.renderer is not None:
                        self.renderer.show_isolated_removal(player, removal['pos'], removal['marble'], task.delay_time)
            elif result.has_captures():
                player.add_capture(result.captured_marbles)

        # Check game over
        game_over = self.session.game.get_game_ended()
        if game_over is not None:  # None means game continuing.
            if game_over in [PLAYER_1_WIN, PLAYER_2_WIN]:
                winner = game_over
            else:
                winner = None  # Tie

            print()
            if winner:
                print(f'Winner: Player {winner}')
            else:
                print('Game ended in a tie')
            self.session.game.print_state()

            print(self.session.player1.captured)
            print(self.session.player2.captured)

            # Increment games played counter
            self.session.increment_games_played()

            if self.session.is_replay_mode():
                print("Replay complete")
                return task.done
            elif self.max_games is not None and self.session.get_games_played() >= self.max_games:
                # Reached max games limit
                print(f"Completed {self.session.get_games_played()} game(s)")
                self._close_log_file()  # Close log file before exiting
                return task.done
            else:
                # Continue with next game
                self._reset_board()
        return task.again