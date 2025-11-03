"""Loader for Boardspace-style SGF transcripts.

Parses Boardspace SGF logs (Boardspace's Zèrtz variant with P0/P1 properties)
into action dictionaries that can be replayed with ReplayZertzPlayer.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Iterable

from sgfmill import sgf

from game.zertz_game import ZertzGame
from game.zertz_board import ZertzBoard


class SGFLoader:
    """Loads and parses Boardspace SGF replay files."""

    def __init__(
        self,
        filename: str | Path,
        status_reporter: Callable[[str], None] | None = None,
    ):
        self.filename = Path(filename)
        self._status_reporter = status_reporter

        self.detected_rings: int | None = None
        self.variant: str | None = None
        self.game_result: str | None = None
        self.player1_name: str | None = None
        self.player2_name: str | None = None
        self.player_times: dict[int, str] = {}
        self.blitz = False
        self.marbles = None
        self.win_condition = None
        self._color_map = {"0": "w", "1": "g", "2": "b"}

    # Public API -----------------------------------------------------------------

    def set_status_reporter(self, reporter: Callable[[str], None] | None) -> None:
        """Set or update the status reporter callback."""
        self._status_reporter = reporter

    def load(self) -> tuple[list[dict], list[dict]]:
        """Load the SGF file and return replay-ready action dictionaries.

        Returns:
            Tuple of lists: (player1_actions, player2_actions). Each list contains
            action dictionaries in the format expected by ReplayZertzPlayer.
        """
        self._report(f"Loading SGF from: {self.filename}")
        raw_text = self.filename.read_text(encoding="utf-8")

        self._extract_player_names(raw_text)
        sanitized = self._sanitize_sgf(raw_text)
        game = sgf.Sgf_game.from_string(sanitized.strip())
        root = game.get_root()

        self.variant = self._get_root_property(root, "SU")
        self.game_result = self._get_root_property(root, "RE")

        turns, coord_samples = self._parse_turns(game)

        if not turns:
            self._report("No moves found in SGF.")
            return [], []

        self.detected_rings = self._detect_board_size(coord_samples)
        self._report(f"Detected board size: {self.detected_rings} rings")

        zgame = ZertzGame(rings=self.detected_rings)
        self.marbles = zgame.marbles
        self.win_condition = zgame.win_con

        player_actions: dict[int, list[dict]] = {1: [], 2: []}

        for player_idx, commands in turns:
            if not commands:
                continue

            has_r_to_b = any(cmd[0] == "RtoB" for cmd in commands)
            has_b_to_b = any(cmd[0] == "BtoB" for cmd in commands)

            if has_r_to_b:
                action = self._process_placement(commands, zgame)
                player_actions[player_idx].append(action)
            elif has_b_to_b:
                captures = self._process_captures(commands, zgame)
                player_actions[player_idx].extend(captures)
            elif any(cmd[0] == "PASS" for cmd in commands):
                player_actions[player_idx].append({"action": "PASS"})
                zgame.take_action("PASS", None)
            else:
                self._report(
                    f"Warning: Unhandled command sequence: {[cmd[0] for cmd in commands]}"
                )

        return player_actions[1], player_actions[2]

    # Internal helpers -----------------------------------------------------------

    def _process_placement(
        self,
        commands: list[list[str]],
        game: ZertzGame,
    ) -> dict:
        """Convert an SGF turn containing placement commands into an action dict."""
        r_to_b = next((cmd for cmd in commands if cmd[0] == "RtoB"), None)
        r_minus = next((cmd for cmd in commands if cmd[0] == "R-"), None)

        if r_to_b is None:
            raise ValueError("Placement turn missing RtoB command")
        if len(r_to_b) < 5:
            raise ValueError(f"Unexpected RtoB format: {r_to_b}")

        dst = self._normalize_coordinate(r_to_b[3], r_to_b[4])
        remove = ""
        if r_minus and len(r_minus) >= 3:
            remove = self._normalize_coordinate(r_minus[1], r_minus[2])

        color_code = r_to_b[2] if len(r_to_b) > 2 else ""
        marble = self._resolve_marble_color(game, dst, remove, color_code)
        action_str = f"PUT {marble} {dst} {remove}".strip()
        action_type, action = game.str_to_action(action_str)

        if action is None:
            raise ValueError(f"Could not translate placement action: {action_str}")

        game.take_action(action_type, action)
        return {
            "action": "PUT",
            "marble": marble,
            "dst": dst,
            "remove": remove,
        }

    def _process_captures(
        self,
        commands: list[list[str]],
        game: ZertzGame,
    ) -> list[dict]:
        """Convert a list of capture commands into action dicts."""
        captures: list[dict] = []
        for cmd in commands:
            if cmd[0] != "BtoB" or len(cmd) < 5:
                self._report(f"Skipping unexpected capture command: {cmd}")
                continue

            src = self._normalize_coordinate(cmd[1], cmd[2])
            dst = self._normalize_coordinate(cmd[3], cmd[4])

            from hiivelabs_mcts import algebraic_to_coordinate
            src_idx = algebraic_to_coordinate(src, game.board.config)
            dst_idx = algebraic_to_coordinate(dst, game.board.config)
            cap_idx = game.board.get_middle_ring(src_idx, dst_idx)
            cap_label = str(game.board.yx_to_label(cap_idx))

            if not cap_label:
                raise ValueError(f"Could not determine capture midpoint for {src}->{dst}")

            moving_color = game.board.get_marble_type_at(src_idx)
            captured_color = game.board.get_marble_type_at(cap_idx)

            action_dict = {
                "action": "CAP",
                "marble": moving_color,
                "src": src,
                "dst": dst,
                "capture": captured_color,
                "cap": cap_label,
            }

            action_str = f"CAP {src} {captured_color or 'b'} {dst}"
            action_type, action = game.str_to_action(action_str)
            if action is None:
                raise ValueError(f"Could not translate capture action: {action_str}")

                # Should never happen, but guard to avoid corrupting state
            game.take_action(action_type, action)
            captures.append(action_dict)
        return captures

    def _resolve_marble_color(
        self,
        game: ZertzGame,
        dst: str,
        remove: str,
        code: str,
    ) -> str:
        """Resolve the marble colour for a placement, updating the colour map if needed."""
        if code and code in self._color_map:
            candidate = self._color_map[code]
            if self._can_apply(game, candidate, dst, remove):
                return candidate
            # Mapping incorrect; drop and re-evaluate
            self._color_map.pop(code, None)

        preferred = [c for c in ("w", "g", "b") if c not in self._color_map.values()]
        fallback = [c for c in ("w", "g", "b") if c in self._color_map.values()]
        candidates = preferred + fallback

        for candidate in candidates:
            if self._can_apply(game, candidate, dst, remove):
                if code:
                    self._color_map[code] = candidate
                return candidate

        raise ValueError(f"Unable to resolve marble color for placement at {dst}")

    def _can_apply(self, game: ZertzGame, marble: str, dst: str, remove: str) -> bool:
        """Check whether placing a marble of the given colour is legal in the current state."""
        action_str = f"PUT {marble} {dst} {remove}".strip()
        action_type, action = game.str_to_action(action_str)
        if action is None:
            return False
        test_game = ZertzGame(clone=game, clone_state=game.board.state)
        try:
            test_game.take_action(action_type, action)
        except ValueError:
            return False
        return True

    def _parse_turns(
        self,
        game: sgf.Sgf_game,
    ) -> tuple[list[tuple[int, list[list[str]]]], list[str]]:
        """Extract the main SGF sequence into per-player command lists."""
        turns: list[tuple[int, list[list[str]]]] = []
        coords: list[str] = []

        current_player: int | None = None
        current_commands: list[list[str]] = []

        for node in game.get_main_sequence()[1:]:
            props = node.properties()
            if not props:
                continue

            prop = props[0]
            raw_value = node.get(prop)
            value = raw_value.decode("utf-8").strip() if isinstance(raw_value, bytes) else raw_value.strip()

            if not value:
                continue

            tokens = value.split()
            if not tokens:
                continue

            if tokens[0].isdigit():
                tokens = tokens[1:]
                if not tokens:
                    continue

            command = tokens[0]
            if command == "Start":
                continue
            if command.lower() == "time":
                player_idx = 1 if prop == "PA" else 2
                self.player_times[player_idx] = tokens[1] if len(tokens) > 1 else ""
                continue
            if command == "Done":
                if current_commands and current_player is not None:
                    turns.append((current_player, current_commands))
                current_commands = []
                current_player = None
                continue

            player_idx = 1 if prop == "PA" else 2
            if current_player is None:
                current_player = player_idx
            elif current_player != player_idx:
                if current_commands:
                    turns.append((current_player, current_commands))
                current_commands = []
                current_player = player_idx

            command_entry = [command] + tokens[1:]
            current_commands.append(command_entry)
            coords.extend(self._extract_coordinates(command_entry))

        if current_commands and current_player is not None:
            turns.append((current_player, current_commands))

        return turns, coords

    def _extract_coordinates(self, command: list[str]) -> list[str]:
        """Return all board coordinates referenced in a command."""
        cmd_type = command[0]
        coords: list[str] = []

        if cmd_type == "RtoB" and len(command) >= 5:
            coords.append(f"{command[3].upper()}{command[4]}")
        elif cmd_type == "R-" and len(command) >= 3:
            coords.append(f"{command[1].upper()}{command[2]}")
        elif cmd_type == "BtoB" and len(command) >= 5:
            coords.append(f"{command[1].upper()}{command[2]}")
            coords.append(f"{command[3].upper()}{command[4]}")

        return [coord for coord in coords if re.match(r"^[A-Z][0-9]+$", coord)]

    def _detect_board_size(self, coords: Iterable[str]) -> int:
        """Infer the board size from the SGF variant metadata."""
        variant = (self.variant or "").lower()
        if "zertz+24" in variant:
            return ZertzBoard.LARGE_BOARD_61
        if "zertz+11" in variant:
            return ZertzBoard.MEDIUM_BOARD_48
        return ZertzBoard.SMALL_BOARD_37

    def _normalize_coordinate(self, letter: str, number: str) -> str:
        """Concatenate the SGF letter and numeric tokens into a board coordinate."""
        digits = "".join(ch for ch in number if ch.isdigit())
        if not digits:
            raise ValueError(f"Invalid coordinate digits: {number}")
        letter_up = letter.upper()
        if self.detected_rings == ZertzBoard.LARGE_BOARD_61 and letter_up == "I":
            letter_up = "J"
        return f"{letter_up}{digits}"

    @staticmethod
    def _sanitize_sgf(raw_text: str) -> str:
        """Replace non-standard property names so sgfmill can parse the tree."""
        return raw_text.replace("P0[", "PA[").replace("P1[", "PB[")

    def _extract_player_names(self, raw_text: str) -> None:
        """Extract player names from SGF text and set player1_name and player2_name."""
        match0 = re.search(r'P0\[id "([^"]+)"]', raw_text)
        match1 = re.search(r'P1\[id "([^"]+)"]', raw_text)
        if match0:
            self.player1_name = match0.group(1)
        if match1:
            self.player2_name = match1.group(1)

    @staticmethod
    def _get_root_property(node: sgf.Sgf_node, prop: str) -> str | None:
        if not node.has_property(prop):
            return None
        values = node.get(prop)
        value = values[0] if isinstance(values, list) and values else values
        if isinstance(value, bytes):
            return value.decode("utf-8")
        return value

    def _report(self, message: str | None) -> None:
        if message is None:
            return
        if self._status_reporter:
            self._status_reporter(message)
        else:
            print(message)
