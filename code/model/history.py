"""Code taken from https://github.com/devsisters/DQN-tensorflow/blob/master/dqn/history.py"""

import numpy as np

from utils.constants import *

class History:
    def __init__(self, config):
        batch_size, history_length, channels = \
            config[BATCH_SIZE], config[HISTORY_LENGTH], config[NUM_CHANNELS]

        self._history = np.zeros(
            [history_length, channels], dtype=np.float32)
    
    @property
    def history(self):
        return self._history

    def add(self, current_price):
        self._history[:-1] = self._history[1:]
        self._history[-1] = current_price

    def reset(self):
        self._history *= 0
