import json
import time
import numpy as np
from game2048 import Game, GameWindow
import NN


def play_game(Q, ui, move_delay):
    game = Game()
    while not game.isGameOver():
        select_and_play_move(Q, game)
        time.sleep(move_delay)
        if ui:
            ui.paint(game)


def select_and_play_move(Q, game):
    """
    determine "best" move and play it. if "best" move is illegal, try next best, etc.
    :param Q: neural network representing value function
    :param game: Game object to play move on
    """

    action_values = Q.forward(game.get_board())

    # try moves until a legal move is found, or all 4 moves fail
    attempts = 0
    action_is_valid = False
    while attempts < 4 and not action_is_valid:
        # try "best" move
        action = action_values.argmax()
        action_is_valid = game.move(action)

        # flag action as invalid so we don't re-attempt it
        action_values[action] = -np.inf
        attempts += 1

        actionmap = ["LEFT", "RIGHT", "UP", "DOWN"]
        print(actionmap[action] + ("" if action_is_valid else " (invalid)"))



def load_network(weights_path, bias_path):
    with open(weights_path, 'r') as f:
        weights = json.load(f)
    with open(bias_path, 'r') as f:
        bias = json.load(f)

    shape = [16] + [len(layer) for layer in bias]
    nn = NN.MultiLayerPerceptron(shape)
    for layer, b, w in zip(nn.layers, bias, weights):
        layer.b = np.array(b)
        layer.w = np.array(w)

    return nn


if __name__ == "__main__":
    move_delay = 0.3  # seconds of sleep between moves

    # load network from disk
    weights_path = "data/nn_weights.json"
    bias_path = "data/nn_bias.json"
    Q = load_network(weights_path, bias_path)

    ui = GameWindow()
    while True:
        play_game(Q, ui, move_delay)
