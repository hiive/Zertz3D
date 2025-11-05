from __future__ import annotations

from game.zertz_game import ZertzGame
from game.players.zertz_player import ZertzPlayer


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
