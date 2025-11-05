from __future__ import annotations

import numpy as np
from hiivelabs_mcts import ZertzAction, get_capture_destination

from game.zertz_game import ZertzGame
from game.players.zertz_player import ZertzPlayer

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
        p1, p2, p3, p4, p5 = p_actions.nonzero()

        # Determine action type
        config = self.game.board.config
        if c1.size > 0:
            # Capture available (and therefore mandatory)
            ip = np.random.randint(c1.size)
            direction, src_y, src_x = int(c1[ip]), int(c2[ip]), int(c3[ip])
            # from game.zertz_board import ZertzBoard
            # action_data = ZertzBoard.capture_indices_to_action(
            #     direction, y, x, self.game.board.config.width, self.game.board.DIRECTIONS
            # )
            # return ("CAP", action_data)
            dst_y, dst_x = get_capture_destination(config, src_y, src_x, direction)
            action = ZertzAction.capture(config, src_y, src_x, dst_y, dst_x)
            return action
        elif p1.size > 0:
            # Only placements available
            ip = np.random.randint(p1.size)
            marble_idx, src_y, src_x, dst_y, dst_x = int(p1[ip]), int(p2[ip]), int(p3[ip]), int(p4[ip]), int(p5[ip])
            action = ZertzAction.placement(self.game.board.config, marble_idx, src_y, src_x, dst_y, dst_x)
            return action
            return ("PUT", (marble_idx, src_y, src_x, dst_y, dst_x))
        else:
            # No valid actions - player must pass
            return ZertzAction.pass_action()

    def get_last_action_scores(self):
        """Random player treats all moves equally (uniform scores).

        Returns:
            Dict mapping action tuples to uniform score of 1.0
        """
        p_actions, c_actions = self.game.get_valid_actions()

        c1, c2, c3 = c_actions.nonzero()
        p1, p2, p3, p4, p5 = p_actions.nonzero()

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
                action = ("PUT", (int(p1[i]), int(p2[i]), int(p3[i]), int(p4[i]), int(p5[i])))
                marble_idx, src_y, src_x, dst_y, dst_x = int(p1[i]), int(p2[i]), int(p3[i]), int(p4[i]), int(p5[i])
                action = ZertzAction.placement(self.game.board.config, marble_idx, src_y, src_x, dst_y, dst_x)
                action_scores[action] = 1.0
        else:
            # Must pass
            action_scores[ZertzAction.pass_action()] = 1.0

        return action_scores
