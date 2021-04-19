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
        #   - for placement actions, shape is 3 x width^2 x (width^2 + 1)
        #   - for capture actions, shape is 6 x width x width
        w = self.game.board.width
        p_actions, c_actions = self.game.get_valid_actions()

        c1, c2, c3 = (c_actions == True).nonzero()
        p1, p2, p3 = (p_actions == True).nonzero()
        if c1.size > 0 and p1.size > 0:
            a_ix = np.random.choice([0, 1])
        elif c1.size > 0:
            a_ix = 1
        else:
            a_ix = 0

        if a_ix == 0:
            a1, a2, a3 = (p1, p2, p3)
            ax = 'PUT'
        else:
            a1, a2, a3 = (c1, c2, c3)
            ax = 'CAP'

        ip = np.random.randint(a1.size)
        action = ax, (a1[ip], a2[ip], a3[ip])
        return action
