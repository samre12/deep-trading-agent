"""Code adapted from https://github.com/tambetm/simple_dqn/blob/master/src/replay_memory.py"""

import random
import numpy as np
from os.path import join

from utils.constants import *
from utils.strings import *
from utils.util import print_and_log_message, print_and_log_message_list

class ReplayMemory:
    '''Memory buffer for experiance replay'''

    def __init__(self, logger, config):
        self.logger = logger

        self._model_dir = join(config[SAVE_DIR], REPLAY_MEMORY)

        self.batch_size = config[BATCH_SIZE]
        self.history_length = config[HISTORY_LENGTH]
        self.memory_size = config[MEMORY_SIZE]
        self.num_channels = config[NUM_CHANNELS]

        self.actions = np.empty(self.memory_size, dtype = np.uint8)
        self.rewards = np.empty(self.memory_size, dtype = np.integer)
        self.screens = np.empty((self.memory_size, config[NUM_CHANNELS]), dtype = np.float32)
        self.terminals = np.empty(self.memory_size, dtype = np.bool)
        
        # pre-allocate prestates and poststates for minibatch
        self.prestates = np.empty((self.batch_size, self.history_length, self.num_channels), 
                                        dtype = np.float32)
        self.poststates = np.empty((self.batch_size, self.history_length, self.num_channels), 
                                        dtype = np.float32)
        
        self.count = 0
        self.current = 0

    def add(self, screen, reward, action, terminal):
        if screen.shape != (self.num_channels):
            print_and_log_message(INVALID_TIMESTEP, self.logger)
        else:
            self.actions[self.current] = action
            self.rewards[self.current] = reward
            self.screens[self.current, ...] = screen
            self.terminals[self.current] = terminal
            self.count = max(self.count, self.current + 1)
            self.current = (self.current + 1) % self.memory_size

    def getState(self, index):
        if self.count == 0:
            print_and_log_message(REPLAY_MEMORY_ZERO, self.logger)
        else:
            index = index % self.count
            if index >= self.history_length - 1:
                return self.screens[(index - (self.history_length - 1)):(index + 1), ...]
            else:
                indexes = [(index - i) % self.count for i in reversed(range(self.history_length))]
                return self.screens[indexes, ...]

    def save(self):
        message = "Saving replay memory to {}".format(self._model_dir)
        for idx, (name, array) in enumerate(
            zip([ACTIONS, REWARDS, SCREENS, TERMINALS, PRESTATES, POSTSTATES],
                [self.actions, self.rewards, self.screens, self.terminals, self.prestates, self.poststates])):
            save_npy(array, join(self._model_dir, name))

        message = "Replay memory successfully saved to {}".format(self._model_dir)
        print_and_log_message(message, self.logger)

    def load(self):
        message = "Loading replay memory from {}".format(self._model_dir)
        print_and_log_message(message, self.logger)

        for idx, (name, array) in enumerate(
            zip([ACTIONS, REWARDS, SCREENS, TERMINALS, PRESTATES, POSTSTATES],
                [self.actions, self.rewards, self.screens, self.terminals, self.prestates, self.poststates])):
            array = load_npy(join(self._model_dir, name))

        message = "Replay memory successfully loaded from {}".format(self._model_dir)
        print_and_log_message(message, self.logger)

    @property
    def model_dir(self):
        return self._model_dir
        
    @property
    def sample(self):
        if self.count <= self.history_length:
            print_and_log_message(REPLAY_MEMORY_INSUFFICIENT, self.logger)
        
        else:
            indexes = []
            while len(indexes) < self.batch_size:
                # find random index 
                while True:
                    # sample one index (ignore states wraping over) 
                    index = random.randint(self.history_length, self.count - 1)
                    # if wraps over current pointer, then get new one
                    if index >= self.current and index - self.history_length < self.current:
                        continue
                    # if wraps over episode end, then get new one
                    # NB! poststate (last screen) can be terminal state!
                    if self.terminals[(index - self.history_length):index].any():
                        continue
                    # otherwise use this index
                    break
                
                # NB! having index first is fastest in C-order matrices
                self.prestates[len(indexes), ...] = self.getState(index - 1)
                self.poststates[len(indexes), ...] = self.getState(index)
                indexes.append(index)

            actions = self.actions[indexes]
            rewards = self.rewards[indexes]
            terminals = self.terminals[indexes]

            return self.prestates, actions, rewards, self.poststates, terminals
