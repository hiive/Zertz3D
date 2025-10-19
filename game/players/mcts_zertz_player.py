from game.zertz_player import ZertzPlayer
from learner.mcts.mcts_tree import MCTSTree
from learner.mcts.transposition_table import TranspositionTable


class MCTSZertzPlayer(ZertzPlayer):
    """MCTS-based Zertz player.

    Uses Monte Carlo Tree Search with UCB1 selection and optional transposition table.
    Drop-in replacement for RandomZertzPlayer with same interface.
    """

    def __init__(self, game, n, iterations=1000, exploration_constant=1.41,
                 max_simulation_depth=None, use_transposition_table=True,
                 use_transposition_lookups=True, time_limit=None, verbose=False,
                 clear_table_each_move=True, parallel='multiprocess', num_workers=16):
        """Initialize MCTS player.

        Args:
            game: ZertzGame instance
            n: Player number (1 or 2)
            iterations: MCTS iterations per move (default: 1000)
            exploration_constant: UCB1 exploration (default: √2 ≈ 1.41)
            max_simulation_depth: Max rollout depth (None = play to end)
            use_transposition_table: Enable symmetry caching (serial mode only)
            use_transposition_lookups: Use cached stats to initialize nodes (serial mode only)
            time_limit: Max search time per move in seconds (None = no limit)
            verbose: Print search statistics after each move
            clear_table_each_move: Clear transposition table between moves (serial mode only)
            parallel: Parallelization mode: False (serial), 'thread' (threaded), 'multiprocess' (default)
            num_workers: Number of threads/processes for parallel search (default: 16)
        """
        super().__init__(game, n)

        self.iterations = iterations
        self.exploration_constant = exploration_constant
        self.max_simulation_depth = max_simulation_depth
        self.use_transposition_lookups = use_transposition_lookups
        self.time_limit = time_limit
        self.verbose = verbose
        self.clear_table_each_move = clear_table_each_move
        self.parallel = parallel
        self.num_workers = num_workers
        self.mcts = MCTSTree()

        # Initialize transposition table (only used in serial mode)
        if use_transposition_table:
            self.transposition_table = TranspositionTable()
        else:
            self.transposition_table = None

    def get_action(self):
        """Select best action using MCTS.

        Returns:
            Action tuple: (action_type, action_data)
        """
        # Optimization: If only one legal move, return it immediately (no search needed)
        placement_mask, capture_mask = self.game.get_valid_actions()

        # Check captures first (they're mandatory)
        c1, c2, c3 = capture_mask.nonzero()
        if c1.size == 1:
            # Exactly one capture - mandatory move, no decision needed
            return ("CAP", (c1[0], c2[0], c3[0]))
        elif c1.size > 1:
            # Multiple captures available - need MCTS to decide
            pass
        else:
            # No captures - check placements
            p1, p2, p3 = placement_mask.nonzero()
            if p1.size == 1:
                # Exactly one placement - forced move, no decision needed
                return ("PUT", (p1[0], p2[0], p3[0]))
            elif p1.size == 0:
                # No moves available - must pass
                return ("PASS", None)
            # Multiple placements available - need MCTS to decide

        # Clear transposition table if configured
        if self.clear_table_each_move and self.transposition_table:
            self.transposition_table.clear()

        # Run MCTS search (multiprocess, threaded, or serial)
        if self.parallel == 'multiprocess':
            action = self.mcts.search_multiprocess(
                self.game,
                iterations=self.iterations,
                exploration_constant=self.exploration_constant,
                max_simulation_depth=self.max_simulation_depth,
                transposition_table=None,  # Not used in multiprocess mode
                use_transposition_lookups=False,
                time_limit=self.time_limit,
                verbose=self.verbose,
                num_processes=self.num_workers
            )
        elif self.parallel == 'thread':
            action = self.mcts.search_parallel(
                self.game,
                iterations=self.iterations,
                exploration_constant=self.exploration_constant,
                max_simulation_depth=self.max_simulation_depth,
                transposition_table=self.transposition_table,
                use_transposition_lookups=self.use_transposition_lookups,
                time_limit=self.time_limit,
                verbose=self.verbose,
                num_threads=self.num_workers
            )
        else:  # Serial mode
            action = self.mcts.search(
                self.game,
                iterations=self.iterations,
                exploration_constant=self.exploration_constant,
                max_simulation_depth=self.max_simulation_depth,
                transposition_table=self.transposition_table,
                use_transposition_lookups=self.use_transposition_lookups,
                time_limit=self.time_limit,
                verbose=self.verbose
            )

        return action