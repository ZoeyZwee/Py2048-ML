import json
from datetime import datetime
import random
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import NN
from game2048 import GameWindow, Game


class StatTracker:
    def __init__(self):
        self.losses = []
        self.scores = []

        self.score10avgs = []
        self.loss10avgs = []

        candidates = ["macosx", "qt5agg", "gtk3agg", "tkagg", "wxagg"]
        for candidate in candidates:
            try:
                plt.switch_backend(candidate)
                print('Using backend: ' + candidate)
                break
            except (ImportError, ModuleNotFoundError):
                pass
        plt.ion()

        self.scorefig, self.scoreax = plt.subplots(2)
        self.lossfig, self.lossax = plt.subplots(2)

    def log(self, score, loss):
        self.losses.append(loss)
        self.scores.append(score)

        self.loss10avgs.append(self.get_last_10_loss_average())
        self.score10avgs.append(self.get_last_10_score_average())

    def get_last_10_score_average(self):
        if not self.scores:
            return 0
        if len(self.scores) < 10:
            return sum(self.scores) / len(self.scores)
        return sum(self.scores[-10:]) / 10

    def get_last_10_loss_average(self):
        if not self.losses:
            return 0
        if len(self.losses) < 10:
            return sum(self.losses) / len(self.losses)
        return sum(self.losses[-10:]) / 10

    def get_latest_loss(self):
        return self.losses[-1]

    def plot(self):
        self.scoreax[0].plot(self.scores, 'bo')
        self.scoreax[0].set_title('Score')
        self.scoreax[1].plot(self.score10avgs, 'bo')
        self.scoreax[1].set_title("Running Average (last 10)")
        self.scorefig.canvas.draw()
        self.scorefig.canvas.flush_events()

        self.lossax[0].plot(self.losses, 'bo')
        self.lossax[0].set_title('Batch Loss')
        self.lossax[1].plot(self.loss10avgs, 'bo')
        self.lossax[1].set_title("Running Average (last 10)")
        self.lossfig.canvas.draw()
        self.lossfig.canvas.flush_events()


@dataclass
class Transition:
    state1: np.ndarray
    action: int
    transition_reward: int
    state2: np.ndarray
    is_terminal: bool  # true if this transition resulted in game-over


class TrainingData:
    def __init__(self):
        self.dat = []

    def append(self, transitions):
        self.dat.extend(transitions)
        if len(self.dat) > 50_000:
            self.dat = self.dat[10_000:]

    def sample(self, batch_size):
        return random.sample(self.dat, batch_size)

    def save(self, path):
        pass

    def load(self, path):
        pass

    def __len__(self):
        return len(self.dat)


def play_game(Q, random_move_chance, illegal_move_penalty, ui):
    game = Game()
    game_is_live = True
    transitions = []
    while game_is_live:
        move_transitions, game_is_live = select_and_play_move(Q, game, random_move_chance, illegal_move_penalty)
        transitions.extend(move_transitions)
        if ui:
            ui.paint(game)

    return game.score, transitions


def select_and_play_move(Q, game, random_move_chance, illegal_move_penalty):
    """
    determine "best" move and play it. if "best" move is illegal, try next best, etc.
    :param Q: neural network representing value function
    :param game: Game object to play move on
    :param random_move_chance: probability of selecting moves in random order
    :param illegal_move_penalty: reward for playing an illegal move
    :return: bool indicating whether legal move was found, list of Transitions from attempted moves
    """

    old_board = game.get_board()
    old_score = game.score

    transitions = []

    # determine order in which to attempt actions
    if np.random.random() < random_move_chance:
        # random action values results in trying moves in random order
        action_values = [0.0, 1.0, 2.0, 3.0]
        random.shuffle(action_values)
        action_values = np.array(action_values)
    else:
        action_values = Q.forward(game.get_board())

    # try moves until a legal move is found, or all 4 moves fail
    attempts = 0
    action_is_valid = False
    while attempts < 4 and not action_is_valid:
        # try "best" move
        action = action_values.argmax()
        action_is_valid = game.move(action)

        # store transition
        transition_reward = game.score - old_score if action_is_valid else illegal_move_penalty
        transitions.append(Transition(old_board, action, transition_reward, game.get_board(), game.isGameOver()))

        # flag action as taken so we don't re-attempt it
        action_values[action] = -np.inf
        attempts += 1

    return transitions, action_is_valid


def train_minibatch(Q_current, Q_target, transitions, learning_rate, future_reward_discount):
    batch_size = len(transitions)

    # desired updates for each layer
    delta_w = []
    delta_b = []

    total_loss = 0
    for t in transitions:
        # compute derivative of loss wrt network output
        current_value = Q_current.forward(t.state1)[t.action]  # value of state-action pair
        if t.is_terminal:
            future_value = 0
        else:
            future_value = Q_target.forward(t.state2).max()  # estimated future reward after taking action

        # compute error according to Bellman Equation
        td_error = current_value - (future_reward_discount * future_value + t.transition_reward)

        # dL/dW = dL/dQ * dQ/dW
        # upstream gradient is: dLdQ if action taken (dQdW = 1), 0 if action not taken (dQdW = 0)
        grad_output = np.zeros_like(Q_current.layers[-1].x)
        grad_output[t.action] = td_error

        # compute desired changes to weights, biases
        delta_w_sample, delta_b_sample = Q_current.backward(grad_output, learning_rate)  # get desired updates

        # add desired changes from sample to total
        if len(delta_w) != len(delta_w_sample):  # init delta_w, delta_b on first iteration
            delta_w = delta_w_sample
            delta_b = delta_b_sample
        else:
            for i in range(len(delta_w)):
                delta_w[i] += delta_w_sample[i]
                delta_b[i] += delta_b_sample[i]

        total_loss += td_error ** 2

    # convert sum change to average change
    for i in range(len(delta_w)):
        delta_w[i] = delta_w[i] / batch_size
        delta_b[i] = delta_b[i] / batch_size

    # apply changes
    Q_current.update_weights_biases(delta_w, delta_b)

    return total_loss / batch_size


if __name__ == "__main__":
    ### PATHS ###
    nn_weights_path = "data/nn_weights.json"
    nn_biases_path = "data/nn_bias.json"
    training_data_write_path = "data/training_data.json"

    ### PARAMETERS AND OPTIONS ###
    plotting = True
    render_every = 50  # 1 out of every X games will be displayed
    save_every = 20
    plot_every = 10

    ### HYPERPARAMETERS ###
    nn_shape = [16, 32, 32, 32, 32, 4]
    random_move_chance = 1 / 100
    illegal_move_penalty = 0
    update_targets_every = 10  # copy Q_current from Q_target every X generations
    batch_size = 1024
    learning_rate = 1 / 1000
    future_reward_discount = 0.996

    np.seterr(over='raise')

    training_data = TrainingData()
    tracker = StatTracker()
    ui = GameWindow()
    Q_current = NN.MultiLayerPerceptron(nn_shape)
    Q_target = Q_current.copy()
    games_played = 0
    while True:
        games_played += 1

        # save
        active_ui = ui if games_played % render_every == 0 else None
        score, transitions = play_game(Q_current, random_move_chance, illegal_move_penalty, active_ui)
        training_data.append(transitions)

        # train and plot loss/scores
        if len(training_data) > batch_size:
            if plotting and games_played % plot_every == 0:
                tracker.plot()

            minibatch_data = training_data.sample(batch_size)
            loss = train_minibatch(Q_current, Q_target, minibatch_data, learning_rate, future_reward_discount)
            tracker.log(score, loss)

        # save NN weights
        if games_played % save_every == 0:
            print("saving network parameters")
            weights = [layer.w.tolist() for layer in Q_current.layers]
            biases = [layer.b.tolist() for layer in Q_current.layers]
            with open(nn_weights_path, 'w') as f:
                json.dump(weights, f)
            with open(nn_biases_path, 'w') as f:
                json.dump(biases, f)

        # update target Q
        if games_played % update_targets_every == 0:
            print("updating targets...")
            Q_target = Q_current.copy()
        print(f"{games_played=}, {len(training_data)=}")
