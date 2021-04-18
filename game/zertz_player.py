# import sys
#
# sys.path.append('..')


class ZertzPlayer:
    def __init__(self, game, n):
        self.captured = {'b': 0, 'g': 0, 'w': 0}
        self.game = game
        self.n = n

    def get_action(self):
        pass


class HumanZertzPlayer(ZertzPlayer):
    pass


class RandomZertzPlayer(ZertzPlayer):
    pass
