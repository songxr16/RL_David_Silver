from enum import Enum


class State:
    def __init__(self, dealer_current_sum, player_current_sum, is_terminal=False):
        self.player_sum = player_current_sum
        self.dealer_sum = dealer_current_sum
        self.terminal = is_terminal
        self.reward = 0


class Card:
    def __init__(self, color, value):
        self.color = color
        self.value = value


class Actions(Enum):
    hit = 0
    stick = 1

    @staticmethod
    def get_action(n):
        if n == 0:
            return Actions.hit
        else:
            return Actions.stick

    @staticmethod
    def get_value(action):
        if action == Actions.hit:
            return 0
        else:
            return 1

    @staticmethod
    def get_values():
        return [0, 1]


class Colors(Enum):
    black = 1
    red = -1
