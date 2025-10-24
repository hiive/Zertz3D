import numpy as np
import math

# Virtual loss constants for parallel MCTS
# Virtual loss temporarily inflates visit counts and adds pessimistic values
# to discouraIt'sge multiple threads from exploring the same path simultaneously
VIRTUAL_LOSS = 3
VIRTUAL_LOSS_VALUE = -3.0  # Pessimistic value


class MCTSNode:
    """Stateless node in the MCTS search tree.

    Stores only state arrays (not game objects) for memory efficiency.
    Memory usage: ~1KB per node vs ~3-5KB for object-based nodes.
    """

    def __init__(self, board_state, global_state, config, canonicalizer,
                 parent=None, action=None, parent_player=None, transposition_table=None):
        """Initialize stateless MCTS node.

        Args:
            board_state: Board state array (will be copied)
            global_state: Global state array (will be copied)
            config: BoardConfig instance (shared, not copied)
            canonicalizer: Canonicalizer for transposition lookups (shared)
            parent: Parent MCTSNode or None for root
            action: Action that led from parent to this node
            parent_player: Player who made the action (for value perspective)
            transposition_table: Optional transposition table for initializing statistics
        """
        # Store state (copy to avoid aliasing)
        self.board_state = board_state.copy()
        self.global_state = global_state.copy()
        self.config = config  # Shared immutable config
        self.canonicalizer = canonicalizer  # Shared canonicalizer

        # Tree structure
        self.parent = parent
        self.action = action
        self.parent_player = parent_player  # Who moved to create this node
        self.children = {}  # action -> MCTSNode
        self.untried_actions = None  # Will be populated on first visit

        self._transposition_table_entry = None
        self._visits = 0
        self._value = 0.0

        if transposition_table is not None:
            entry = transposition_table.get_entry(
                self.board_state,
                self.global_state,
                self.canonicalizer,
                config=self.config,
                create=True,
            )
            if entry is not None:
                self._transposition_table_entry = entry

    @property
    def visits(self):
        if self._transposition_table_entry is not None:
            return self._transposition_table_entry['visits']
        return self._visits

    @visits.setter
    def visits(self, value):
        if self._transposition_table_entry is not None:
            self._transposition_table_entry['visits'] = value
        else:
            self._visits = value

    @property
    def value(self):
        if self._transposition_table_entry is not None:
            return self._transposition_table_entry['value']
        return self._value

    @value.setter
    def value(self, value):
        if self._transposition_table_entry is not None:
            self._transposition_table_entry['value'] = value
        else:
            self._value = value

    def count_legal_actions(self):
        """Count the number of legal actions from this state.

        Matches Rust implementation for consistency.
        """
        from game import zertz_logic

        p_actions, c_actions = zertz_logic.get_valid_actions(
            self.board_state, self.global_state, self.config
        )

        placement_count = np.count_nonzero(p_actions)
        capture_count = np.count_nonzero(c_actions)
        total = placement_count + capture_count

        # At least 1 for PASS if no other actions
        return max(total, 1)

    def is_fully_expanded(self, widening_constant=None):
        """Check if node should be expanded further.

        With progressive widening enabled (widening_constant is not None), limits
        children based on parent visits: max_children = widening_constant * sqrt(parent_visits)

        This focuses search on promising moves instead of trying every legal move.
        With large branching factors (Zertz has 100+ moves early game), this is
        essential to get enough visits per move to distinguish good from bad.

        Args:
            widening_constant: Controls branching factor (None = disabled, 10.0 = moderate)

        Returns:
            True if no more children should be added
        """
        children_count = len(self.children)
        legal_actions = self.count_legal_actions()

        if widening_constant is None:
            # Standard MCTS: expand until all actions tried
            return children_count == legal_actions

        # Progressive widening: limit children based on visit count
        # max_children grows with sqrt(visits), focusing search
        max_children = int(widening_constant * math.sqrt(self.visits + 1))
        max_children = min(max_children, legal_actions)

        return children_count >= max_children

    def is_terminal(self):
        """Check if this is a terminal game state.

        Checks both standard terminal conditions (win, board full, no marbles)
        and consecutive passes (both players must pass = draw).
        """
        from game import zertz_logic

        # Check standard terminal conditions
        if zertz_logic.is_game_over(self.board_state, self.global_state, self.config):
            return True

        # Check for consecutive passes using parent pointer
        # If current node was created by a PASS and current player must also PASS,
        # then both players passed consecutively → terminal (draw)
        if self.parent is not None and self.action == ("PASS", None):
            # Parent passed to get here, check if current player must also pass
            p_actions, c_actions = zertz_logic.get_valid_actions(
                self.board_state, self.global_state, self.config
            )

            current_must_pass = (c_actions.sum() == 0 and p_actions.sum() == 0)
            if current_must_pass:
                return True  # Consecutive passes → draw

        return False

    def _populate_untried_actions(self):
        """Populate list of untried actions from current game state."""
        if self.is_terminal():
            self.untried_actions = []
            return

        # Use stateless functions to get valid actions
        from game import zertz_logic

        p_actions, c_actions = zertz_logic.get_valid_actions(
            self.board_state, self.global_state, self.config
        )

        # Convert action masks to list of action tuples
        actions = []

        # Check for captures (mandatory if available)
        c1, c2, c3 = c_actions.nonzero()
        if c1.size > 0:
            for i in range(c1.size):
                actions.append(("CAP", (c1[i], c2[i], c3[i])))
        else:
            # Check for placements
            p1, p2, p3 = p_actions.nonzero()
            if p1.size > 0:
                for i in range(p1.size):
                    actions.append(("PUT", (p1[i], p2[i], p3[i])))
            else:
                # No valid actions - must pass
                actions.append(("PASS", None))

        self.untried_actions = actions

    def get_untried_action(self):
        """Get a random untried action."""
        if self.untried_actions is None:
            self._populate_untried_actions()

        if len(self.untried_actions) == 0:
            return None

        # Random selection
        idx = np.random.randint(len(self.untried_actions))
        return self.untried_actions.pop(idx)

    def add_virtual_loss(self):
        """Add virtual loss to discourage thread collision.

        Virtual loss temporarily inflates visit count and adds pessimistic value
        to make this path less attractive to other threads during parallel search.
        Must be paired with remove_virtual_loss() after backpropagation.

        NOTE: Currently not used in Python implementation due to multiprocessing
        (separate memory spaces). Reserved for future use with Python 3.13+ free-threaded
        mode or Python 3.14+ where true shared-memory multithreading becomes practical.
        Included for API parity with Rust implementation.
        """
        self.visits = self.visits + VIRTUAL_LOSS
        self.value = self.value + VIRTUAL_LOSS_VALUE

    def remove_virtual_loss(self):
        """Remove virtual loss after backpropagation.

        Removes the temporary inflation added by add_virtual_loss().
        Should be called before adding the real simulation value.

        NOTE: Currently not used in Python implementation. See add_virtual_loss()
        docstring for details.
        """
        self.visits = self.visits - VIRTUAL_LOSS
        self.value = self.value - VIRTUAL_LOSS_VALUE
