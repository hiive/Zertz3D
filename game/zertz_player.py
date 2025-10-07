# import sys
#
# sys.path.append('..')
from game.zertz_game import ZertzGame
import numpy as np


class ZertzPlayer:
    def __init__(self, game: ZertzGame, n):
        self.captured = {'b': 0, 'g': 0, 'w': 0}
        self.game = game
        self.n = n

    def get_action(self):
        pass

    def add_capture(self, capture):
        self.captured[capture] += 1


class HumanZertzPlayer(ZertzPlayer):
    pass


class RandomZertzPlayer(ZertzPlayer):

    def get_action(self):
        """
        Select random valid action.
        - Placement actions: shape (3, width², width² + 1)
        - Capture actions: shape (6, width, width)
        Note: Captures are mandatory, so placement mask will be empty if captures exist.
        """
        p_actions, c_actions = self.game.get_valid_actions()

        c1, c2, c3 = (c_actions == True).nonzero()
        p1, p2, p3 = (p_actions == True).nonzero()

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
            # No valid actions (shouldn't happen in a valid game state)
            raise ValueError("No valid actions available")

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
