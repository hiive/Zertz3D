# import sys
#
# sys.path.append('..')
from __future__ import annotations

from typing import Optional

import queue

from game.zertz_game import ZertzGame
import numpy as np
from shared.constants import MARBLE_TYPES


class ZertzPlayer:
    """Base player with shared state and lifecycle hooks."""

    def __init__(self, game: ZertzGame, n):
        self.captured = {'b': 0, 'g': 0, 'w': 0}
        self.game = game
        self.n = n

    def get_action(self):
        raise NotImplementedError

    #
    # Lifecycle hooks
    #
    def on_turn_start(
        self,
        context: str,
        placement_mask: Optional[np.ndarray],
        capture_mask: Optional[np.ndarray],
    ) -> None:
        """Inform the player that a new decision context has begun."""
        return None

    def clear_context(self) -> None:
        """Signal that the current decision context is complete."""
        return None

    def add_capture(self, capture):
        self.captured[capture] += 1


class HumanZertzPlayer(ZertzPlayer):
    def __init__(self, game: ZertzGame, n):
        super().__init__(game, n)
        self._action_queue: queue.Queue = queue.Queue()
        self._latest_masks: dict[str, Optional[np.ndarray]] = {
            'placement': None,
            'capture': None,
        }
        self._current_context: Optional[str] = None
        self._current_options: Optional[dict] = None

    def get_action(self):
        return self._action_queue.get()

    def submit_action(self, action):
        self._action_queue.put(action)

    def cancel_pending_action(self):
        try:
            while True:
                self._action_queue.get_nowait()
        except queue.Empty:
            pass

    def set_context_masks(self, placement_mask: np.ndarray, capture_mask: np.ndarray) -> None:
        self._latest_masks['placement'] = np.array(placement_mask, copy=True)
        self._latest_masks['capture'] = np.array(capture_mask, copy=True)

    def clear_context_masks(self) -> None:
        self._latest_masks['placement'] = None
        self._latest_masks['capture'] = None

    def get_context_masks(self) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        return self._latest_masks['placement'], self._latest_masks['capture']

    def pending_actions_empty(self) -> bool:
        return self._action_queue.empty()

    def get_current_options(self) -> Optional[dict]:
        return self._current_options

    def _compute_options(
        self,
        placement_mask: Optional[np.ndarray],
        capture_mask: Optional[np.ndarray],
    ) -> Optional[dict]:
        if placement_mask is None and capture_mask is None:
            return None

        board = self.game.board
        width = board.width

        options: dict = {
            'context': self._current_context,
            'placement': set(),
            'removal': set(),
            'capture_sources': set(),
            'capture_destinations': set(),
            'capture_paths': {},
            'supply_colors': set(),
        }

        if capture_mask is not None and np.any(capture_mask):
            for direction in range(capture_mask.shape[0]):
                ys, xs = np.where(capture_mask[direction])
                for y, x in zip(ys, xs):
                    src = (int(y), int(x))
                    options['capture_sources'].add(src)
                    dy, dx = board.DIRECTIONS[direction]
                    cap_index = (y + dy, x + dx)
                    dst_index = board.get_jump_destination(src, cap_index)
                    if dst_index is None:
                        continue
                    dst_y, dst_x = dst_index
                    if not (0 <= dst_y < width and 0 <= dst_x < width):
                        continue
                    if board.state[board.RING_LAYER, dst_y, dst_x] != 1:
                        continue
                    dst = (int(dst_y), int(dst_x))
                    options['capture_destinations'].add(dst)
                    options['capture_paths'].setdefault(src, set()).add(dst)

        if placement_mask is not None and np.any(placement_mask):
            for marble_idx in range(placement_mask.shape[0]):
                put_flat, rem_flat = np.where(placement_mask[marble_idx])
                if put_flat.size > 0:
                    options['supply_colors'].add(MARBLE_TYPES[marble_idx])
                for put, rem in zip(put_flat, rem_flat):
                    put_y, put_x = divmod(int(put), width)
                    if board.state[board.RING_LAYER, put_y, put_x] == 1:
                        options['placement'].add((put_y, put_x))
                    if rem != width ** 2:
                        rem_y, rem_x = divmod(int(rem), width)
                        if board.state[board.RING_LAYER, rem_y, rem_x] == 1:
                            options['removal'].add((rem_y, rem_x))

        return options

    def on_turn_start(
        self,
        context: str,
        placement_mask: Optional[np.ndarray],
        capture_mask: Optional[np.ndarray],
    ) -> None:
        """Cache incoming masks and reset pending input for a new decision phase."""
        context_changed = context != self._current_context
        self._current_context = context
        if context_changed:
            self.cancel_pending_action()
        if placement_mask is not None:
            self._latest_masks['placement'] = np.array(placement_mask, copy=True)
        else:
            self._latest_masks['placement'] = None
        if capture_mask is not None:
            self._latest_masks['capture'] = np.array(capture_mask, copy=True)
        else:
            self._latest_masks['capture'] = None
        self._current_options = self._compute_options(
            self._latest_masks['placement'],
            self._latest_masks['capture'],
        )

    def clear_context(self) -> None:
        """Clear cached context and masks once the phase completes."""
        self.cancel_pending_action()
        self._current_context = None
        self.clear_context_masks()
        self._current_options = None

    def current_context(self) -> Optional[str]:
        return self._current_context


class RandomZertzPlayer(ZertzPlayer):

    def get_action(self):
        """
        Select random valid action.
        - Placement actions: shape (3, width², width² + 1)
        - Capture actions: shape (6, width, width)
        Note: Captures are mandatory, so placement mask will be empty if captures exist.

        If no valid actions exist, player passes ('PASS', None).
        """
        p_actions, c_actions = self.game.get_valid_actions()

        c1, c2, c3 = c_actions.nonzero()
        p1, p2, p3 = p_actions.nonzero()

        # Determine action type
        if c1.size > 0:
            # Capture available (and therefore mandatory)
            ax = 'CAP'
            a1, a2, a3 = c1, c2, c3
        elif p1.size > 0:
            # Only placements available
            ax = 'PUT'
            a1, a2, a3 = p1, p2, p3
        else:
            # No valid actions - player must pass
            return ('PASS', None)

        ip = np.random.randint(a1.size)
        action = ax, (a1[ip], a2[ip], a3[ip])
        return action


class ReplayZertzPlayer(ZertzPlayer):
    """Player that replays moves from a list of action dictionaries."""

    def __init__(self, game: ZertzGame, n, actions):
        super().__init__(game, n)
        self.actions = actions
        self.action_index = 0

    def get_action(self):
        """Return the next action from the replay list."""
        if self.action_index >= len(self.actions):
            raise ValueError(f"No more actions for player {self.n}")

        action_dict = self.actions[self.action_index]
        self.action_index += 1

        # Convert action_dict to action string format
        if action_dict['action'] == 'PUT':
            # Check if remove field exists and is not empty
            if action_dict.get('remove') and action_dict['remove'].strip():
                action_str = f"PUT {action_dict['marble']} {action_dict['dst']} {action_dict['remove']}"
            else:
                action_str = f"PUT {action_dict['marble']} {action_dict['dst']}"
        elif action_dict['action'] == 'CAP':
            action_str = f"CAP {action_dict['marble']} {action_dict['src']} {action_dict['capture']} {action_dict['dst']}"
        else:
            raise ValueError(f"Unknown action type: {action_dict['action']}")

        # Use game's str_to_action to convert to internal format
        return self.game.str_to_action(action_str)
