from utils.constants import *
from utils.strings import *

class BaseAgent(object):
    '''Base class containing all the parameters for reinforcement learning'''

    def __init__(self, config):
        scale = 10000

        self.max_step = 5000 * scale

        self.target_q_update_step = 1 * scale
        self.learning_rate = 0.00025
        self.learning_rate_minimum = 0.00025
        self.learning_rate_decay = 0.96
        self.learning_rate_decay_step = 5 * scale

        self.ep_end = 0.1
        self.ep_start = 1.
        self.ep_end_t = config[MEMORY_SIZE]

        self.train_frequency = 4
        self.learn_start = 5. * scale

        self.min_delta = -1
        self.max_delta = 1

        # _test_step = 5 * scale
        # _save_step = _test_step * 10