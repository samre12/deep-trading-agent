"""Code adapted from https://github.com/devsisters/DQN-tensorflow/tree/master/dqn/replay_memory.py"""

import random
import numpy as np
from os.path import join

from model.util import save_npy, load_npy

from utils.constants import *
from utils.strings import *

class ReplayMemory:
    '''Memory buffer for experiance replay'''

    def __init__(self, logger, config):
        self.logger = logger

        self._model_dir = join(config[SAVE_DIR], REPLAY_MEMORY)

        self.batch_size = config[BATCH_SIZE]
        self.history_length = config[HISTORY_LENGTH]
        self.memory_size = config[MEMORY_SIZE]
        self.num_channels = config[NUM_CHANNELS]
        self.dims = (self.num_channels,)

        self.actions = np.empty(self.memory_size, dtype = np.uint8)
        self.rewards = np.empty(self.memory_size, dtype = np.float32)
        self.screens = np.empty((self.memory_size, config[NUM_CHANNELS]), dtype = np.float32)
        self.terminals = np.empty(self.memory_size, dtype = np.bool)
        self.trades_rem = np.empty(self.memory_size, dtype = np.float32)
        
        # pre-allocate prestates and poststates for minibatch
        self.prestates = (np.empty((self.batch_size, self.history_length, self.num_channels), 
                                        dtype = np.float32),\
                                        np.empty(self.batch_size, dtype=np.float32))
        self.poststates = (np.empty((self.batch_size, self.history_length, self.num_channels), 
                                        dtype = np.float32),\
                                        np.empty(self.batch_size, dtype=np.float32))
        
        self.count = 0
        self.current = 0

    def add(self, screen, reward, action, terminal, trade_rem):
        if screen.shape != self.dims:
            self.logger.error(INVALID_TIMESTEP)
            
        else:
            self.actions[self.current] = action
            self.rewards[self.current] = reward
            self.screens[self.current, ...] = screen
            self.terminals[self.current] = terminal
            self.trades_rem[self.current] = trade_rem
            self.count = max(self.count, self.current + 1)
            self.current = (self.current + 1) % self.memory_size

    def getState(self, index):
        if self.count == 0:
            self.logger.error(REPLAY_MEMORY_ZERO)
            
        else:
            index = index % self.count
            if index >= self.history_length - 1:
                return self.screens[(index - (self.history_length - 1)):(index + 1), ...], \
                        self.trades_rem[index]
                        
            else:
                indexes = [(index - i) % self.count for i in reversed(range(self.history_length))]
                return self.screens[indexes, ...], self.trade_rem[index]

    def save(self):
        message = "Saving replay memory to {}".format(self._model_dir)
        self.logger.info(message)
        for idx, (name, array) in enumerate(
            zip([ACTIONS, REWARDS, SCREENS, TERMINALS, TRADES_REM, PRESTATES, POSTSTATES],
                [self.actions, self.rewards, self.screens, self.terminals, self.trades_rem, self.prestates, self.poststates])):
            save_npy(array, join(self._model_dir, name))

        message = "Replay memory successfully saved to {}".format(self._model_dir)
        self.logger.info(message)

    def load(self):
        message = "Loading replay memory from {}".format(self._model_dir)
        self.logger.info(message)

        for idx, (name, array) in enumerate(
            zip([ACTIONS, REWARDS, SCREENS, TERMINALS, TRADES_REM, PRESTATES, POSTSTATES],
                [self.actions, self.rewards, self.screens, self.terminals, self.trades_rem, self.prestates, self.poststates])):
            array = load_npy(join(self._model_dir, name))

        message = "Replay memory successfully loaded from {}".format(self._model_dir)
        self.logger.info(message)

    @property
    def model_dir(self):
        return self._model_dir
        
    @property
    def sample(self):
        if self.count <= self.history_length:
            self.logger.error(REPLAY_MEMORY_INSUFFICIENT)
        
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
                self.prestates[0][len(indexes), ...], self.prestates[0][len(indexes)] = self.getState(index - 1)
                self.poststates[0][len(indexes), ...], self.poststates[1][len(indexes)] = self.getState(index)
                indexes.append(index)

            actions = self.actions[indexes]
            rewards = self.rewards[indexes]
            terminals = self.terminals[indexes]

            return self.prestates, actions, rewards, self.poststates, terminals
