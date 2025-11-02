# import sys
#
# sys.path.append('..')
from __future__ import annotations

from typing import Optional, Any

import queue

from game.zertz_game import ZertzGame
import numpy as np
from shared.constants import MARBLE_TYPES


class ZertzPlayer:
    """Base player with shared state and lifecycle hooks."""

    def __init__(self, game: ZertzGame, n):
        self.captured = {"b": 0, "g": 0, "w": 0}
        self.game = game
        self.n = n
        self.name = f"Player {n}"

    def get_action(self):
        raise NotImplementedError

    def get_last_action_scores(self):
        """Get normalized scores for all legal actions from last search.

        Returns:
            Dict mapping action tuples to normalized scores [0.0, 1.0]
        """
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

    def handle_selection(self, selection: dict) -> bool:
        """Handle renderer-driven selection events (default: ignore)."""
        return False

    def handle_hover(self, selection: Optional[dict]) -> bool:
        return False

    def add_capture(self, capture):
        self.captured[capture] += 1


class HumanZertzPlayer(ZertzPlayer):
    def __init__(self, game: ZertzGame, n):
        super().__init__(game, n)
        self._action_queue: queue.Queue = queue.Queue()
        self._latest_masks: dict[str, Optional[np.ndarray]] = {
            "placement": None,
            "capture": None,
        }
        self._current_context: Optional[str] = None
        self._current_options: Optional[dict] = None
        self._selection_state: dict[str, Any] = {}

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

    def set_context_masks(
        self, placement_mask: np.ndarray, capture_mask: np.ndarray
    ) -> None:
        self._latest_masks["placement"] = np.array(placement_mask, copy=True)
        self._latest_masks["capture"] = np.array(capture_mask, copy=True)

    def clear_context_masks(self) -> None:
        self._latest_masks["placement"] = None
        self._latest_masks["capture"] = None

    def get_context_masks(self) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        return self._latest_masks["placement"], self._latest_masks["capture"]

    def pending_actions_empty(self) -> bool:
        return self._action_queue.empty()

    def get_current_options(self) -> Optional[dict]:
        return self._current_options

    def get_selection_state(self) -> dict:
        snapshot = dict(self._selection_state)
        last = snapshot.get("last")
        if isinstance(last, dict):
            snapshot["last"] = dict(last)
        placement_pending = snapshot.get("placement_pending_removals")
        if isinstance(placement_pending, set):
            snapshot["placement_pending_removals"] = set(placement_pending)
        return snapshot

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
            "context": self._current_context,
            "placement": set(),
            "removal": set(),
            "capture_sources": set(),
            "capture_destinations": set(),
            "capture_paths": {},
            "supply_colors": set(),
            "supply_counts": tuple(
                int(x) for x in board.global_state[board.SUPPLY_SLICE]
            ),
        }
        options["supply_total"] = sum(options["supply_counts"])

        if capture_mask is not None and np.any(capture_mask):
            for direction in range(capture_mask.shape[0]):
                ys, xs = np.where(capture_mask[direction])
                for y, x in zip(ys, xs):
                    src = (int(y), int(x))
                    options["capture_sources"].add(src)
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
                    options["capture_destinations"].add(dst)
                    options["capture_paths"].setdefault(src, set()).add(dst)

        if placement_mask is not None and np.any(placement_mask):
            for marble_idx in range(placement_mask.shape[0]):
                put_flat, rem_flat = np.where(placement_mask[marble_idx])
                if put_flat.size > 0:
                    options["supply_colors"].add(MARBLE_TYPES[marble_idx])
                for put, rem in zip(put_flat, rem_flat):
                    put_y, put_x = divmod(int(put), width)
                    if board.state[board.RING_LAYER, put_y, put_x] == 1:
                        options["placement"].add((put_y, put_x))
                    if rem != width**2:
                        rem_y, rem_x = divmod(int(rem), width)
                        if board.state[board.RING_LAYER, rem_y, rem_x] == 1:
                            options["removal"].add((rem_y, rem_x))

        return options

    def handle_selection(self, selection: dict) -> bool:
        if not self.pending_actions_empty():
            return False

        options = self._current_options
        if not options:
            return False

        context = options.get("context")
        board = self.game.board

        if context == "capture":
            return self._handle_capture_selection(selection, options, board)
        elif context == "placement":
            return self._handle_placement_selection(selection, options, board)

        return False

    def _handle_placement_selection(
        self, selection: dict, options: dict, board
    ) -> bool:
        selection_type = selection.get("type")
        width = board.width
        placement_mask = self._latest_masks.get("placement")
        if placement_mask is None:
            return False

        state = self._selection_state

        if selection_type == "supply_marble":
            color = selection.get("color")
            if color not in options["supply_colors"]:
                return False
            if color not in MARBLE_TYPES:
                return False
            color_idx = MARBLE_TYPES.index(color)
            if options["supply_counts"][color_idx] <= 0:
                return False
            state["placement_color"] = color
            state["placement_color_idx"] = color_idx
            state["placement_source"] = "supply"
            if "supply_key" in selection:
                state["placement_supply_key"] = selection["supply_key"]
            state.pop("placement_dst", None)
            state.pop("placement_dst_flat", None)
            state.pop("placement_pending_removals", None)
            state.pop("placement_removal_index", None)
            state.pop("placement_allow_none_removal", None)
            state.pop("last", None)
            return True

        if selection_type == "captured_marble":
            if options.get("supply_total", 0) > 0:
                return False
            color = selection.get("color")
            owner = selection.get("owner")
            if (
                owner != self.n
                or color not in options["supply_colors"]
                or color not in MARBLE_TYPES
            ):
                return False
            state["placement_color"] = color
            state["placement_color_idx"] = MARBLE_TYPES.index(color)
            state["placement_source"] = "captured"
            if "captured_key" in selection:
                state["placement_captured_key"] = selection["captured_key"]
            state.pop("placement_dst", None)
            state.pop("placement_dst_flat", None)
            state.pop("placement_pending_removals", None)
            state.pop("placement_removal_index", None)
            state.pop("placement_allow_none_removal", None)
            state.pop("last", None)
            return True

        if selection_type not in ("ring", "board_marble"):
            return False

        index = selection.get("index")
        if index is None:
            return False
        idx_tuple = (int(index[0]), int(index[1]))

        color_idx = state.get("placement_color_idx")
        if color_idx is None:
            return False

        dest_flat = idx_tuple[0] * width + idx_tuple[1]

        # If removal is pending and this ring is one of them, treat as removal selection
        pending_removals = state.get("placement_pending_removals")
        if pending_removals:
            allow_none = state.get("placement_allow_none_removal", False)
            current_dst = state.get("placement_dst")
            if allow_none and current_dst == idx_tuple:
                return self._finalize_placement(
                    color_idx, state.get("placement_dst_flat"), width**2
                )
            if idx_tuple in pending_removals:
                removal_flat = idx_tuple[0] * width + idx_tuple[1]
                return self._finalize_placement(
                    color_idx,
                    dest_flat=state.get("placement_dst_flat"),
                    removal_flat=removal_flat,
                )

        # Otherwise treat as destination selection
        if dest_flat >= placement_mask.shape[1]:
            return False

        valid_removals = np.where(placement_mask[color_idx, dest_flat])[0]
        if valid_removals.size == 0:
            return False

        state["placement_dst"] = idx_tuple
        state["placement_dst_flat"] = dest_flat

        pending = {
            (int(r // width), int(r % width)) for r in valid_removals if r != width**2
        }
        state["placement_pending_removals"] = pending
        state["placement_allow_none_removal"] = (width**2) in valid_removals

        if len(valid_removals) == 1:
            removal_flat = int(valid_removals[0])
            return self._finalize_placement(color_idx, dest_flat, removal_flat)

        if (width**2) in valid_removals and not pending:
            return self._finalize_placement(color_idx, dest_flat, width**2)

        state["last"] = {"type": selection_type, "index": idx_tuple}
        return True

    def _finalize_placement(
        self, color_idx: int, dest_flat: Optional[int], removal_flat: Optional[int]
    ) -> bool:
        if dest_flat is None or removal_flat is None:
            return False
        self.submit_action(("PUT", (color_idx, dest_flat, removal_flat)))
        self._selection_state = {}
        return True

    def _handle_capture_selection(self, selection: dict, options: dict, board) -> bool:
        selection_type = selection.get("type")
        capture_mask = self._latest_masks.get("capture")
        if capture_mask is None:
            return False

        state = self._selection_state
        width = board.width

        if selection_type == "board_marble":
            index = selection.get("index")
            if index is None:
                return False
            idx_tuple = (int(index[0]), int(index[1]))
            if idx_tuple not in options["capture_sources"]:
                return False
            state["capture_src"] = idx_tuple
            state["last"] = {"type": "board_marble", "index": idx_tuple}
            return True

        if selection_type != "ring":
            return False

        src = state.get("capture_src")
        if src is None:
            return False

        index = selection.get("index")
        if index is None:
            return False
        dst = (int(index[0]), int(index[1]))
        if dst not in options["capture_paths"].get(src, set()):
            return False

        action = self._resolve_capture_action(src, dst, capture_mask, board)
        if action is None:
            return False
        self.submit_action(("CAP", action))
        self._selection_state = {}
        return True

    def handle_hover(self, selection: Optional[dict]) -> bool:
        options = self._current_options
        if options is None:
            self._selection_state.pop("hover", None)
            return False

        if not selection:
            if "hover" in self._selection_state:
                self._selection_state.pop("hover", None)
                return True
            return False

        context = options.get("context")
        selection_type = selection.get("type")
        hover_entry: dict[str, Any] = {"type": selection_type}

        if selection_type in ("ring", "board_marble"):
            index = selection.get("index")
            if index is None:
                self._selection_state.pop("hover", None)
                return False
            idx_tuple = (int(index[0]), int(index[1]))
            valid = False
            board = self.game.board
            pending = self._selection_state.get("placement_pending_removals")
            if context == "placement":
                if pending:
                    valid = idx_tuple in pending
                else:
                    color_idx = self._selection_state.get("placement_color_idx")
                    placement_mask = self._latest_masks.get("placement")
                    if placement_mask is None or color_idx is None:
                        valid = idx_tuple in options.get("placement", set())
                    else:
                        width = board.width
                        dest_flat = idx_tuple[0] * width + idx_tuple[1]
                        if 0 <= dest_flat < placement_mask.shape[1]:
                            valid = bool(np.any(placement_mask[color_idx, dest_flat]))
            elif context == "capture":
                if selection_type == "board_marble":
                    valid = idx_tuple in options.get("capture_sources", set())
                else:
                    src = self._selection_state.get("capture_src")
                    if src is None:
                        valid = idx_tuple in options.get("capture_sources", set())
                    valid = valid or idx_tuple in options.get(
                        "capture_destinations", set()
                    )
            if not valid:
                self._selection_state.pop("hover", None)
                return False
            hover_entry["index"] = idx_tuple
        elif selection_type == "supply_marble":
            color = selection.get("color")
            if color not in options.get("supply_colors", set()):
                self._selection_state.pop("hover", None)
                return False
            hover_entry["color"] = color
            if "supply_key" in selection:
                hover_entry["supply_key"] = selection["supply_key"]
        elif selection_type == "captured_marble":
            color = selection.get("color")
            owner = selection.get("owner")
            if (
                options.get("supply_total", 0) > 0
                or owner != self.n
                or color not in options.get("supply_colors", set())
            ):
                self._selection_state.pop("hover", None)
                return False
            hover_entry["color"] = color
            hover_entry["owner"] = owner
            if "captured_key" in selection:
                hover_entry["captured_key"] = selection["captured_key"]
        else:
            self._selection_state.pop("hover", None)
            return False

        if self._selection_state.get("hover") != hover_entry:
            self._selection_state["hover"] = hover_entry
            return True
        return False

    def _resolve_capture_action(
        self,
        src: tuple[int, int],
        dst: tuple[int, int],
        capture_mask: np.ndarray,
        board,
    ) -> Optional[tuple[None, int, int]]:
        y, x = src
        for direction in range(capture_mask.shape[0]):
            if not capture_mask[direction, y, x]:
                continue
            dy, dx = board.DIRECTIONS[direction]
            cap_index = (y + dy, x + dx)
            computed_dst = board.get_jump_destination(src, cap_index)
            if computed_dst == dst:
                # Convert to (None, src_flat, dst_flat) format
                src_flat = y * board.width + x
                dst_y, dst_x = dst
                dst_flat = dst_y * board.width + dst_x
                return (None, src_flat, dst_flat)
        return None

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
            self._selection_state = {}
        if placement_mask is not None:
            self._latest_masks["placement"] = np.array(placement_mask, copy=True)
        else:
            self._latest_masks["placement"] = None
        if capture_mask is not None:
            self._latest_masks["capture"] = np.array(capture_mask, copy=True)
        else:
            self._latest_masks["capture"] = None
        self._current_options = self._compute_options(
            self._latest_masks["placement"],
            self._latest_masks["capture"],
        )

    def clear_context(self) -> None:
        """Clear cached context and masks once the phase completes."""
        self.cancel_pending_action()
        self._current_context = None
        self.clear_context_masks()
        self._current_options = None
        self._selection_state = {}

    def current_context(self) -> Optional[str]:
        return self._current_context


class RandomZertzPlayer(ZertzPlayer):

    def __init__(self, game: ZertzGame, n):
        super().__init__(game, n)
        self.name = f"Random {n}"
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
            ip = np.random.randint(c1.size)
            direction, y, x = int(c1[ip]), int(c2[ip]), int(c3[ip])
            from game.zertz_board import ZertzBoard
            action_data = ZertzBoard.capture_indices_to_action(
                direction, y, x, self.game.board.width, self.game.board.DIRECTIONS
            )
            return ("CAP", action_data)
        elif p1.size > 0:
            # Only placements available
            ip = np.random.randint(p1.size)
            return ("PUT", (int(p1[ip]), int(p2[ip]), int(p3[ip])))
        else:
            # No valid actions - player must pass
            return ("PASS", None)

    def get_last_action_scores(self):
        """Random player treats all moves equally (uniform scores).

        Returns:
            Dict mapping action tuples to uniform score of 1.0
        """
        p_actions, c_actions = self.game.get_valid_actions()

        c1, c2, c3 = c_actions.nonzero()
        p1, p2, p3 = p_actions.nonzero()

        action_scores = {}

        # Collect all valid actions with uniform score
        if c1.size > 0:
            # Captures available
            for i in range(c1.size):
                action = ("CAP", (int(c1[i]), int(c2[i]), int(c3[i])))
                action_scores[action] = 1.0
        elif p1.size > 0:
            # Placements available
            for i in range(p1.size):
                action = ("PUT", (int(p1[i]), int(p2[i]), int(p3[i])))
                action_scores[action] = 1.0
        else:
            # Must pass
            action_scores[("PASS", None)] = 1.0

        return action_scores


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
        if action_dict["action"] == "PUT":
            # Check if remove field exists and is not empty
            if action_dict.get("remove") and action_dict["remove"].strip():
                action_str = f"PUT {action_dict['marble']} {action_dict['dst']} {action_dict['remove']}"
            else:
                action_str = f"PUT {action_dict['marble']} {action_dict['dst']}"
        elif action_dict["action"] == "CAP":
            action_str = f"CAP {action_dict['src']} {action_dict['capture']} {action_dict['dst']}"
        else:
            raise ValueError(f"Unknown action type: {action_dict['action']}")

        # Use game's str_to_action to convert to internal format
        return self.game.str_to_action(action_str)
