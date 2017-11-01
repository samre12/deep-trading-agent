"""Code taken from https://github.com/devsisters/DQN-tensorflow/blob/master/dqn/history.py"""

import numpy as np

from utils.constants import *
from utils.util import print_and_log_message

class History:
    '''Experiance buffer of the behaniour policy of the agent'''

    def __init__(self, logger, config):
        self.logger = logger

        batch_size, history_length, self.num_channels = \
            config[BATCH_SIZE], config[HISTORY_LENGTH], config[NUM_CHANNELS]

        self._history = np.zeros(
            [history_length, self.num_channels], dtype=np.float32)

    def add(self, screen):
        if screen.shape != (self.num_channels):
            print_and_log_message(INVALID_TIMESTEP, self.logger)
            
        self._history[:-1] = self._history[1:]
        self._history[-1] = screen
    
    def set_history(self, history):
        if history.shape != self._history:
            print_and_log_message(INVALID_HISTORY, self.logger)
        
        self._history = history

    @property
    def history(self):
        return self._history
