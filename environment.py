from classes import State, Actions, Colors, Card
from agent import Agent
import numpy as np
import random


class Environment:

    def __init__(self):
        self.card_min = 1
        self.card_max = 10
        self.dealer_values = 10
        self.player_values = 21
        self.action_values = len(Actions.get_values())

    def get_initial_state(self):
        dealer_first_card = self.draw_black_card()
        player_first_card = self.draw_black_card()
        return State(dealer_first_card.value, player_first_card.value)

    def round(self, state, action):
        if action == Actions.hit:
            card = self.draw_card()
            if card.color == Colors.black:
                value = card.value
            else:
                value = -card.value
            next_state = State(state.dealer_sum, state.player_sum+value)
            if next_state.player_sum < 1 or next_state.player_sum > 21:
                next_state.terminal = True
                next_state.reward = -1
        else:
            next_state = State(state.dealer_sum, state.player_sum)
            draw_again = True
            while draw_again:
                card = self.draw_card()
                if card.color == Colors.black:
                    value = card.value
                else:
                    value = -card.value
                next_state.dealer_sum = next_state.dealer_sum+value
                if next_state.dealer_sum < 1 or next_state.dealer_sum > 21:
                    next_state.reward = 1
                    draw_again = False
                elif next_state.dealer_sum > 16:
                    if next_state.dealer_sum < next_state.player_sum:
                        next_state.reward = 1
                    elif next_state.dealer_sum == next_state.player_sum:
                        next_state.reward = 0
                    else:
                        next_state.reward = -1
                    draw_again = False
            next_state.terminal = True
        return next_state

    def draw_card(self):
        if random.random() < 2.0/3.0:
            return self.draw_black_card()
        else:
            return Card(Colors.red, random.randint(self.card_min, self.card_max))

    def draw_black_card(self):
        return Card(Colors.black, random.randint(self.card_min, self.card_max))
