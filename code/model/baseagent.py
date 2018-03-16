"""Code adapted from https://github.com/devsisters/DQN-tensorflow/tree/master/dqn/base.py"""

import os

import tensorflow as tf

from utils.constants import *
from utils.strings import *

class BaseAgent(object):
    '''Base class containing all the parameters for reinforcement learning'''

    def __init__(self, config, logger):
        self.logger = logger
        self._checkpoint_dir = os.path.join(config[SAVE_DIR], 'checkpoints/')
        if not os.path.exists(self._checkpoint_dir):
            os.makedirs(self._checkpoint_dir)

        scale = 5000 #value mentioned originally is 10000

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

        self.min_reward = -1.0
        self.max_reward = 1.0

        self.min_delta = -1
        self.max_delta = 1

        self.test_step = 5 * scale
        self.save_step = self.test_step * 10

        self.env_name = "btc_sim"

    @property
    def checkpoint_dir(self):
        return self._checkpoint_dir

    @property
    def saver(self):
        if self._saver == None:
            self._saver = tf.train.Saver(max_to_keep=10)
        return self._saver

    def save_model(self, step=None):
        message = "Saving checkpoint to {}".format(self.checkpoint_dir)
        self.logger.info(message)
        self.saver.save(self.sess, self.checkpoint_dir, global_step=step)

    def load_model(self):
        message = "Loading checkpoint from {}".format(self.checkpoint_dir)
        self.logger.info(message)

        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            fname = os.path.join(self.checkpoint_dir, ckpt_name)
            self.saver.restore(self.sess, fname)
            message = "Checkpoint successfully loaded from {}".format(fname)
            self.logger.info(message)
            return True
        else:
            message = "Checkpoint could not be loaded from {}".format(self.checkpoint_dir)
            self.logger.info(message)
            return False
