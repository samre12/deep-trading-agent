from __future__ import print_function
import os
import time
import random
import tensorflow as tf
import numpy as np

from model.deepsense import DeepSense
from model.deepsenseparams import DeepSenseParams
from model.history import History

from utils.constants import *
from utils.strings import *
from utils.util import print_and_log_message, print_and_log_message_list

class Agent:
    '''Deep Trading Agent based on Deep Q Learning'''

    def __init__(self, sess, logger, config):
        self.sess = sess
        self.logger = logger
        self.config = config
        params = DeepSenseParams(self.config)

        self.history = History(self.config)

        with tf.variable_scope(STEPS):
            self.step_op = tf.Variable(0, trainable=False, name=STEP)
            self.step_input = tf.placeholder('int32', None, name=STEP_INPUT)
            self.step_assign_op = self.step_op.assign(self.step_input)

        self.q_net = DeepSense(params, logger, sess, name=Q_NETWORK)
        self.t_q_net = DeepSense(params, logger, sess, name=T_Q_NETWORK)

