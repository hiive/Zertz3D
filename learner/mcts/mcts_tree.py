import math
import time

import numpy as np

from learner.mcts.mcts_node import MCTSNode


class MCTSTree:
    """Stateless MCTS tree using pure functions for game logic."""

    @staticmethod
    def ucb1_score(child, parent, exploration_constant):
        """Calculate UCB1 score for child node selection.

        UCB1 balances exploitation (high average value) with exploration
        (less-visited nodes). The exploration_constant (typically √2 ≈ 1.41)
        controls the exploration/exploitation trade-off.

        Args:
            child: Child node to score
            parent: Parent node
            exploration_constant: Exploration parameter (typically 1.41)

        Returns:
            UCB1 score (higher is better)
        """
        if child.visits == 0:
            return float('inf')  # Prioritize unvisited nodes

        # Exploitation term: average value from parent's perspective
        # Note: child.value is from child's player perspective (due to flipping in backprop)
        # We negate to convert to parent's player perspective (opponent of child's player)
        exploitation = -(child.value / child.visits)

        # Exploration term: encourages trying less-visited nodes
        exploration = exploration_constant * math.sqrt(
            math.log(parent.visits) / child.visits
        )

        return exploitation + exploration

    def select(self, node, exploration_constant=1.41):
        """Selection phase: traverse tree using UCB1 until reaching expandable node.

        Args:
            node: Starting node (typically root)
            exploration_constant: UCB1 exploration parameter

        Returns:
            Node to expand (either has untried actions or is terminal)
        """
        while not node.is_terminal() and node.is_fully_expanded():
            # Select best child using UCB1
            best_score = -float('inf')
            best_child = None

            for child in node.children.values():
                score = self.ucb1_score(child, node, exploration_constant)
                if score > best_score:
                    best_score = score
                    best_child = child

            node = best_child

        return node

    @staticmethod
    def expand(node, transposition_table=None):
        """Expansion phase: create new child node for an untried action.

        Uses stateless functions for state transitions.

        Args:
            node: Node to expand from
            transposition_table: Optional transposition table for initializing child statistics

        Returns:
            New child node, or node itself if terminal or no untried actions
        """
        if node.is_terminal():
            return node

        action = node.get_untried_action()
        if action is None:
            return node  # No untried actions (shouldn't happen in normal flow)

        # Import stateless logic
        from game import stateless_logic

        # Get current player before applying action
        current_player = int(node.global_state[node.config.cur_player])

        # Apply action using stateless functions
        action_type, action_data = action
        if action_type == "PASS":
            # PASS just toggles player
            new_board_state = node.board_state.copy()
            new_global_state = node.global_state.copy()
            new_global_state[node.config.cur_player] = 1 - current_player
        else:
            new_board_state, new_global_state = stateless_logic.apply_action(
                node.board_state,
                node.global_state,
                action_data,
                action_type,
                node.config
            )

        # Create child node (optionally initialize from transposition table)
        child = MCTSNode(
            new_board_state,
            new_global_state,
            node.config,
            node.canonicalizer,
            parent=node,
            action=action,
            parent_player=current_player,
            transposition_table=transposition_table
        )

        node.children[action] = child
        return child

    def simulate(self, node, max_depth=None):
        """Simulation phase: play out game randomly from current state.

        Uses stateless functions for efficiency.

        Args:
            node: Node to simulate from
            max_depth: Maximum simulation depth (None = play to end)

        Returns:
            Result from node's current player's perspective: +1 (win), -1 (loss), 0 (draw)
        """
        # Evaluate from the leaf node's current player perspective
        # This is the standard MCTS approach: evaluate from current position,
        # then flip values during backpropagation as we go up the tree
        leaf_player = int(node.global_state[node.config.cur_player])

        if node.is_terminal():
            # Game already over
            return self.evaluate_terminal(node, leaf_player)

        # Import stateless logic
        from game import stateless_logic

        # Clone state for simulation
        sim_board_state = node.board_state.copy()
        sim_global_state = node.global_state.copy()

        depth = 0
        consecutive_passes = 0  # Track consecutive passes to detect infinite loops

        # Play randomly until terminal or max depth
        while True:
            # Check terminal conditions using stateless function
            if stateless_logic.is_game_over(sim_board_state, sim_global_state, node.config):
                return self.evaluate_terminal_game(sim_board_state, sim_global_state,
                                                   node.config, leaf_player)

            # Check for consecutive passes (both players passed)
            if consecutive_passes >= 2:
                # Both players passed - game is a draw (immobilization)
                return 0

            if max_depth is not None and depth >= max_depth:
                # Hit depth limit, use heuristic evaluation
                return self.evaluate_heuristic(sim_board_state, sim_global_state,
                                              node.config, leaf_player)

            # Get random action using stateless functions
            p_actions, c_actions = stateless_logic.get_valid_actions(
                sim_board_state, sim_global_state, node.config
            )

            c1, c2, c3 = c_actions.nonzero()
            p1, p2, p3 = p_actions.nonzero()

            if c1.size > 0:
                # Capture available
                idx = np.random.randint(c1.size)
                action_type = "CAP"
                action_data = (c1[idx], c2[idx], c3[idx])
            elif p1.size > 0:
                # Placement available
                idx = np.random.randint(p1.size)
                action_type = "PUT"
                action_data = (p1[idx], p2[idx], p3[idx])
            else:
                # Must pass
                action_type = "PASS"
                action_data = None

            # Apply action
            if action_type == "PASS":
                consecutive_passes += 1
                current_player = int(sim_global_state[node.config.cur_player])
                sim_global_state[node.config.cur_player] = 1 - current_player
            else:
                consecutive_passes = 0  # Reset pass counter
                sim_board_state, sim_global_state = stateless_logic.apply_action(
                    sim_board_state, sim_global_state, action_data, action_type, node.config
                )

            depth += 1

    def evaluate_terminal(self, node, root_player=None):
        """Evaluate terminal node from root player's perspective.

        Args:
            node: Terminal node to evaluate
            root_player: Player (0 or 1) from whose perspective to evaluate

        Returns:
            +1 if root_player won, -1 if lost, 0 if draw
        """
        if root_player is None:
            root_player = int(node.global_state[node.config.cur_player])
        return self.evaluate_terminal_game(node.board_state, node.global_state,
                                          node.config, root_player)

    @staticmethod
    def evaluate_terminal_game(board_state, global_state, config, root_player):
        """Evaluate terminal game state (stateless version).

        Args:
            board_state: Terminal board state
            global_state: Terminal global state
            config: BoardConfig
            root_player: Player (0 or 1) from whose perspective to evaluate

        Returns:
            +1 if root_player won, -1 if lost, 0 if draw
        """
        from game import stateless_logic

        # Use stateless function to get outcome
        outcome = stateless_logic.get_game_outcome(board_state, global_state, config)

        if outcome is None:
            return 0  # Not terminal (shouldn't happen in simulation)

        # Outcome: 1 (P1 win), -1 (P2 win), or 0 (tie)
        # Convert to root_player's perspective
        if outcome == 0:
            return 0  # Tie
        elif outcome == 1:
            return 1 if root_player == 0 else -1  # P1 won
        elif outcome == -1:
            return 1 if root_player == 1 else -1  # P2 won
        else:
            return 0  # Shouldn't happen

    @staticmethod
    def evaluate_heuristic(board_state, global_state, config, root_player):
        """Heuristic evaluation for non-terminal states (early termination).

        Simple heuristic based on captured marbles:
        - Black marbles: 3 points
        - Gray marbles: 2 points
        - White marbles: 1 point

        Args:
            board_state: Current board state
            global_state: Current global state
            config: BoardConfig
            root_player: Player (0 or 1) from whose perspective to evaluate

        Returns:
            Estimated value in [-1, 1] range
        """
        # Get capture counts for both players from global_state
        p0_captures = global_state[config.p1_cap_slice]  # [white, gray, black]
        p1_captures = global_state[config.p2_cap_slice]

        # Weight by marble value
        weights = np.array([1, 2, 3])  # white, gray, black
        p0_score = np.dot(p0_captures, weights)
        p1_score = np.dot(p1_captures, weights)

        # Calculate advantage
        if root_player == 0:
            advantage = p0_score - p1_score
        else:
            advantage = p1_score - p0_score

        # Normalize to [-1, 1] range
        # Max possible score difference is roughly 30-40 points
        return np.tanh(advantage / 10.0)

    @staticmethod
    def backpropagate(node, result, transposition_table=None):
        """Backpropagation phase: update statistics up the tree.

        Updates both tree nodes and transposition table. Value is flipped
        at each level since players alternate (zero-sum game).

        Args:
            node: Leaf node where simulation ended
            result: Simulation result from leaf node's perspective
            transposition_table: TranspositionTable to update (optional)
        """
        while node is not None:
            node.visits += 1
            node.value += result

            # Update transposition table if provided
            if transposition_table is not None:
                transposition_table.update(node.board_state, node.global_state, node.canonicalizer,
                                         visits_delta=1, value_delta=result)

            # Flip result for parent (opponent's perspective)
            result = -result
            node = node.parent

    def search(self, game, iterations, exploration_constant=1.41,
               max_simulation_depth=None, transposition_table=None,
               use_transposition_lookups=True, time_limit=None, verbose=False):
        """Run MCTS search from current game state.

        Args:
            game: ZertzGame instance to search from
            iterations: Number of MCTS iterations to run
            exploration_constant: UCB1 exploration parameter (default: √2)
            max_simulation_depth: Max simulation depth (None = play to end)
            transposition_table: TranspositionTable for caching (optional but recommended)
            use_transposition_lookups: Use cached statistics to initialize nodes (default: True)
            time_limit: Maximum search time in seconds (None = no limit)
            verbose: Print search statistics

        Returns:
            Best action found, or None if no legal actions
        """
        # Import stateless logic
        from game import stateless_logic

        # Get config from game
        config = game.board._get_config()

        # Only use lookups if both table exists and lookups enabled
        lookup_table = transposition_table if use_transposition_lookups else None

        # Create root node from game state
        root = MCTSNode(
            game.board.state,
            game.board.global_state,
            config,
            game.board.canonicalizer,
            transposition_table=lookup_table
        )

        # Check if any actions available
        if root.is_terminal():
            return "PASS", None

        start_time = time.time()

        for iteration in range(iterations):
            # Check time limit
            if time_limit is not None and time.time() - start_time > time_limit:
                if verbose:
                    print(f"Time limit reached at iteration {iteration}")
                break

            # Selection
            node = self.select(root, exploration_constant)

            # Expansion
            if not node.is_terminal():
                node = self.expand(node, transposition_table=lookup_table)

            # Simulation - evaluate from leaf node's perspective
            result = self.simulate(node, max_simulation_depth)

            # Backpropagation (always update table if provided, regardless of lookup setting)
            self.backpropagate(node, result, transposition_table)

        # Select best move (highest visit count)
        if len(root.children) == 0:
            return ("PASS", None)

        best_action = max(root.children.items(),
                          key=lambda item: item[1].visits)[0]

        if verbose:
            elapsed = time.time() - start_time
            actual_iterations = root.visits
            print(f"\nMCTS Search Statistics:")
            print(f"  Iterations: {actual_iterations}")
            print(f"  Time: {elapsed:.2f}s")
            print(f"  Iterations/sec: {actual_iterations / elapsed:.1f}")
            print(f"  Root visits: {root.visits}")
            print(f"  Root value: {root.value:.2f}")
            print(f"  Children explored: {len(root.children)}")

            if transposition_table:
                hit_rate = transposition_table.get_hit_rate()
                print(f"  Transposition hit rate: {hit_rate:.1%}")

            # Show top 3 moves
            sorted_children = sorted(root.children.items(),
                                     key=lambda item: item[1].visits,
                                     reverse=True)
            print(f"\n  Top moves:")
            for i, (action, child) in enumerate(sorted_children[:3], 1):
                avg_value = child.value / child.visits if child.visits > 0 else 0
                print(f"    {i}. {action[0]}: {child.visits} visits, "
                      f"value {avg_value:.3f}")

        return best_action