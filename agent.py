import random

from convnet import Network
import numpy as np


class Agent:
    def __init__(self, training, nn_architecture, game, rewards, alpha, alpha_decay, gamma, epsilon, epsilon_decay, batch_size, Q=None, D=None, version=0):

        # game stuff
        self.game = game
        self.Q = Network(nn_architecture) if Q is None else Q
        self.Q_target = self.Q.copy()
        self.moves_played = 0

        # learning stuff
        self.training = training
        self.D = [] if D is None else D
        self.rewards = rewards
        self.alpha = alpha               # learning rate
        self.alpha_decay = alpha_decay   # schedule on how to slow alpha
        self.gamma = gamma
        self.batch_size = batch_size

        # epsilon stuff
        self.epsilon = epsilon
        self.epsilon_steps_max = 152
        self.epsilon_steps_taken = 17 # todo: what is this again?
        self.epsilon_decay = epsilon_decay  # unused
        self.maxtile_ever = 2
        # self.boring_storage_rate = boring_storage_rate

        # trackers
        self.version = version
        self.scores = []
        self.losses = []
        self.average_losses_cumulative = []
        self.legals = 0
        self.illegals = 0

    def game_over(self):
        # update trackers
        score = self.game.getscore()
        self.scores.append([score, self.legals, self.illegals])
        self.legals = 0
        self.illegals = 0

        # trim
        def trim(x, threshold, dropsize):
            if len(x) > threshold:
                del x[:dropsize]
        for ls in [self.scores, self.losses, self.average_losses_cumulative]:
            trim(ls, 10000, 1000)
        trim(self.D, self.batch_size*24, self.batch_size*4)

        # take one step of training, provided we have sufficient data
        if self.training and len(self.D) > self.batch_size:
            random.shuffle(self.D)
            self.minibatch(self.D[:self.batch_size])



    def update_target(self):
        self.Q_target = self.Q.copy()

    def play_move(self, legals=None, **kwargs):
        """
        determine "best" move, play move, store training example, then train (if training).
        if "best" move is illegal, try next best, etc.
        :param training: True if training, False if testing
        :param legals: legality of each move [LEFT, RIGHT, UP, DOWN].
        :return: True if game over after move, False otherwise
        """

        legals = [True, True, True, True] if legals is None else legals
        isGameOver = False

        # store old state
        state_1 = self.game.getboard()

        # determine action
        if self.training and np.random.random() > self.epsilon:  # not random move
            action_values = self.Q.forward(self.game.getboard())
            # action = action_values.argmin()
            action = action_values.argmax()
        else:  # random move
            print("it's the ol {:.2f}% chance to do something dumb".format(self.epsilon*100))
            action_values = self.Q.forward(self.game.getboard())
            action = np.random.randint(4)

        while legals[action] is False:
            # run through known illegals to find next best, which might still illegal
            action_values[action] = np.NINF
            # action = action_values.argmin()
            action = action_values.argmax()

            if np.all(action_values == np.array([np.NINF, np.NINF, np.NINF, np.NINF])):
                # game is over, we just didnt know.
                if self.training:
                    self.D[-4][17] = self.game.getscore()  # update transition reward of move that lost the game
                isGameOver = True
                break

        self.legals += 1

        # play move
        conditionmap = {True: 'valid', False: 'invalid'} # used to convert output of game.move() to a string
        condition = conditionmap[self.game.move(action)]
        transition_reward = self.rewards[condition]
        # if action was illegal, flag is as such, and try again, assuming we aren't in game-over state
        if condition == 'invalid' and not isGameOver:
            legals[action] = False
            isGameOver = self.play_move(legals)
            self.illegals += 1

        # get new state
        state_2 = self.game.getboard()

        if self.training:
            # store move in training data
            self.D.append(np.concatenate((state_1, [action, transition_reward], state_2)))
            #maxtile = self.game.matrix.max()
            # if maxtile > self.maxtile_ever:
            #     self.maxtile_ever = maxtile
            #     self.epsilon_steps_taken += 17-(maxtile-1)
            #     self.epsilon = 1-self.epsilon_steps_taken/self.epsilon_steps_max

            # training moved to gameover(), so that the games are fast, and then we train in between them

            # take one step of training, provided we have sufficient data
            # if len(self.D) > self.batch_size*4:
            #     random.shuffle(self.D)
            #     self.minibatch(self.D[:self.batch_size])
        return isGameOver

    def minibatch(self, samples):
        """
        Train on given minibatch of samples
        :param samples: list of training samples (packed)
        :return: none
        """

        # get delta_w, delta_b average over samples
        delta_w = []  # average changes to weights. each entry is a numpy vector with desired changes for 1 layer
        delta_b = []  # average changes to biases. each entry is a numpy vector with desired changes for 1 layer

        avg_loss = 0
        samplecount = 1
        for sample in samples:
            def unpack(x): return x[0:16], int(x[16:17][0]), x[17:18][0], x[18:34]  # quick lil unpack fn
            state_1, action, transition_reward, state_2 = unpack(sample)  # [state_1, action, transition_reward, state_2]

            s1 = state_1.reshape(4,4)
            s2 = state_2.reshape(4,4)
            # compute derivative of loss wrt network output
            Q_out = self.Q.forward(state_1)[action]       # Q of action taken

            Q_target_out = self.Q_target.forward(state_2).max()    # Q of "best" option in new state
            # Q_target_out = self.Q_target.forward(state_2).min()    # Q of "least bad" option in new state

            # TD error
            dLdQ = Q_out - self.gamma*Q_target_out + transition_reward

            # dL/dW = dL/dQ * dQ/dW
            # upstream gradient is: dLdQ if action taken (dQdW = 1), 0 if action not taken (dQdW = 0)
            grad_output = np.zeros_like(self.Q.layers[-1].x)
            grad_output[action] = dLdQ

            # compute desired changes to weights, biases
            delta_w_sample, delta_b_sample = self.Q.backward(grad_output, self.alpha)  # get desired updates

            # update summed desired changes
            if len(delta_w) != len(delta_w_sample):  # init delta_w, delta_b on first iteration
                delta_w = delta_w_sample
                delta_b = delta_b_sample
            else:
                for i in range(len(delta_w)):
                    delta_w[i] += delta_w_sample[i]
                    delta_b[i] += delta_b_sample[i]

            # average loss for debugging purposes
            avg_loss = avg_loss*samplecount
            avg_loss += dLdQ**2
            samplecount += 1
            avg_loss = avg_loss/samplecount

        # convert sum change to average change
        for i in range(len(delta_w)):
            delta_w[i] = delta_w[i] / self.batch_size
            delta_b[i] = delta_b[i] / self.batch_size

        # apply changes
        self.Q.update_weights_biases(delta_w, delta_b)

        # decay alpha and epsilon
        #self.alpha = self.alpha*(1-self.alpha_decay)
        #self.epsilon = self.epsilon*(1-self.epsilon_decay)

        # update trackers
        running_average_loss = 0 if len(self.average_losses_cumulative)==0 else self.average_losses_cumulative[-1]*self.version
        self.version += 1
        self.losses.append(avg_loss)
        self.average_losses_cumulative.append((running_average_loss+avg_loss)/self.version)

    def getHypers(self):
        return self.alpha, self.alpha_decay, self.epsilon*100, self.epsilon_decay
        # return self.alpha, self.alpha_decay, self.epsilon * 100, self.epsilon_steps_taken