import sys

import numpy as np
import matplotlib.pyplot as plt

from game2048 import GameHandler
from agent import Agent

import time
from tkinter import *

"""
    REWARD SCHEME:
        - reward equal to score on loss.
"""

"""    
    ALPHA SCHEDULE:
        alpha < 1
        
        given infinite learning time, the sum over all alpha is unbounded.
        given infinite learning time, the sum over all alpha SQUARED is bounded (convergent).
        i.e.
        lim(t->infinity)[SUM(alpha_t)] = infinity
        lim(t->infinity)[SUM((alpha_t)**2)] < infinity
        

"""


# TODO: DONE implement backprop so as to determine gradient
# TODO: DONE implement learning update for single training example
# TODO: DONE implement cached storage (D)
# TODO: DONE implement minibatching
# TODO: DONE allow network to interface with game - 0=left, 1=right, 2=up, 3=down
# TODO: DONE figure out how to get live visuals
# TODO: DONE clean up network construction
# TODO: DONE RELU
# TODO: DONE validate network
# TODO: DONE epsilon decay
# TODO: DONE alpha decay

# TODO: more frequent rewards??? network learning for a long time then randomly diverging
# TODO: convolutional layers??

class Main:
    nn_architecture = [
        {"length_input": 0, "length": 16, "activation": "input"},

        {"length_input": 16, "length": 16, "activation": "relu"},
        {"length_input": 16, "length": 16, "activation": "relu"},
        {"length_input": 16, "length": 16, "activation": "relu"},
        {"length_input": 16, "length": 16, "activation": "relu"},

        {"length_input": 16, "length": 4, "activation": "relu"},
    ]

    def __init__(self, rewards, options, hyper, plots):
        # options
        self.moves_per_second = options['moves_per_second']
        self.target_update_frequency = options['target_update_frequency']
        self.training = options['training']
        self.plots = plots

        # init game and agent objects
        self.rewards = rewards
        self.game = GameHandler()
        self.agent = Agent(self.training, self.nn_architecture, self.game, rewards, **hyper)

        try:
            self.load(options)
        except FileNotFoundError:
            pass


        # play/train
        stepcount = 0

        # self.buildtestset()
        # self.trainingtest()

        while True:
            if self.agent.play_move():  # playMove() scores true on game over
                self.game_over()
                if self.training and stepcount > self.target_update_frequency:
                    print("Saving network, updating targets...")
                    self.agent.update_target()
                    self.save()
                    stepcount = 0
                stepcount += 1

            if self.moves_per_second > 0:
                time.sleep(1 / self.moves_per_second)
            self.game.paint()

    def buildtestset(self):
        stepcount = 0
        while stepcount < 128 or len(self.agent.scores) < 1:
            if self.agent.play_move(training=self.training):
                self.game_over()
            self.game.paint()
            stepcount += 1

    def trainingtest(self):
        stepcount = 0
        while True:
            if self.training and stepcount > self.target_update_frequency:
                print("updating targets...")
                self.agent.update_target()
                self.plot(**self.plots)

                stepcount = 0
            self.agent.minibatch(self.agent.D)

            stepcount += 1

    def game_over(self):
        self.agent.game_over()

        if self.training:
            self.plot(**self.plots)

        print(
            "game complete. score:{},  legal moves:{} , illegal moves:{}, success rate:{:.2f}%  || training set size:{}, network iteration:{}, games completed:{}".format(
                self.agent.scores[-1][0], self.agent.scores[-1][1], self.agent.scores[-1][2],
                100 * self.agent.scores[-1][1] / (self.agent.scores[-1][1] + self.agent.scores[-1][2]),
                len(self.agent.D), self.agent.version, len(self.agent.scores))
        )

        if self.training:
            print("alpha:{:.2e} ({:.2e}), epsilon:{:.2f}% ({:.2e})".format(*self.agent.getHypers()))

        #if len(self.agent.scores) == 1000:
            #sys.exit("1000 games played")

        self.game.newgame()

    def save(self):
        np.save("NN_version", np.array([self.agent.version]))
        np.save("NN_scores", np.array(self.agent.scores))
        np.save("NN_losses", np.array(self.agent.losses))
        np.save("NN_cum_avg_loss", np.array(self.agent.average_losses_cumulative))
        np.save("NN_zTraining Data", np.stack(self.agent.D))

        for i in range(1, len(self.agent.Q.layers)):
            np.save("NN_weights {}".format(i), self.agent.Q.layers[i].w)
            np.save("NN_biases {}".format(i), self.agent.Q.layers[i].b)

    def load(self, options):
        try:
            if options['load_network']:
                print("loading network...")
                for i in range(1, len(self.agent.Q.layers)):
                    self.agent.Q.layers[i].w = np.load("NN_weights {}.npy".format(i))
                    self.agent.Q.layers[i].b = np.load("NN_biases {}.npy".format(i))
                self.agent.version = np.load("NN_version.npy")[0]
                self.agent.losses = np.load("NN_losses.npy").tolist()
                self.agent.average_losses_cumulative = np.load("NN_cum_avg_loss.npy").tolist()
                self.agent.scores = np.load("NN_scores.npy").tolist()
                self.agent.Q_target = self.agent.Q.copy()

            if options['load_training_data']:
                print("loading training data...")
                self.agent.D = [row for row in np.load("NN_zTraining Data.npy")]

            for i in range(self.agent.version):
                self.agent.epsilon = self.agent.epsilon / (1 + self.agent.epsilon_decay)
                self.agent.alpha = self.agent.alpha / (1 + self.agent.alpha_decay)

        except FileNotFoundError:
            pass

    def plot(self, **kwargs):
        """
        :param args[0]: plot scores
        :param args[1]: plot cost/loss
        :param args[2]: plot move success rate
        :return:
        """
        show = False

        if kwargs['plotscores']:
            fig = scorefig
            axs = scoreax
            axs.plot([score[0] for score in self.agent.scores], 'bo')
            axs.set_title('Score')
            fig.canvas.draw()
            fig.canvas.flush_events()

        if kwargs['plotsuccess']:
            fig = successfig
            axs = successax
            # self.agent.scores is list of (score, valid_count, invalid_count)
            axs[0].plot([score[1] + score[2] for score in self.agent.scores], 'bo')
            axs[0].set_title('Moves Played (including invalid)')
            axs[1].plot([score[1] / (score[1] + score[2]) for score in self.agent.scores], 'bo')
            axs[1].set_title('Move Success Rate')
            axs[1].set_ylim(0.7, 0.9)
            fig.canvas.draw()
            fig.canvas.flush_events()

        if kwargs['plotloss'] and self.training:
            fig = lossfig
            axs = lossax
            axs[0].plot(self.agent.losses, 'bo')
            axs[0].set_title('Average Loss - Single Batch')
            axs[1].plot(self.agent.average_losses_cumulative, 'bo')
            axs[1].set_title('Average Loss - Cumulative')
            fig.canvas.draw()
            fig.canvas.flush_events()


if __name__ == "__main__":
    np.seterr(over='raise')

    # make pyplot print to external backend
    candidates = ["macosx", "qt5agg", "gtk3agg", "tkagg", "wxagg"]
    for candidate in candidates:
        try:
            plt.switch_backend(candidate)
            print('Using backend: ' + candidate)
            break
        except (ImportError, ModuleNotFoundError):
            pass

    plt.ion()

    scorefig, scoreax = plt.subplots(1)
    lossfig, lossax = plt.subplots(2)
    successfig, successax = plt.subplots(2)


    __options = dict(load_network=False, load_training_data=True, training=True, moves_per_second=0,
                     target_update_frequency=30)
    __hyperparams = dict(alpha=1e-5, alpha_decay=1e-10, gamma=0.5, epsilon=0.0000, epsilon_decay=0, batch_size=4096)
    __rewards = dict(invalid=-.001, valid=0, loss=-10)
    __plots = dict(plotscores=True, plotloss=True, plotsuccess=True)
    Main(__rewards, __options, __hyperparams, __plots)
