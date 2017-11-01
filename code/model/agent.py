"""Code adapted from https://github.com/tambetm/simple_dqn/blob/master/src/replay_memory.py"""

from __future__ import print_function
import os
import time
import random
import tensorflow as tf
import numpy as np

from model.baseagent import BaseAgent
from model.deepsense import DeepSense
from model.deepsenseparams import DeepSenseParams
from model.environment import Environment
from model.history import History
from model.replay_memory import ReplayMemory

from utils.constants import *
from utils.strings import *
from utils.util import print_and_log_message, print_and_log_message_list, \
                        clipped_error

class Agent(BaseAgent):
    '''Deep Trading Agent based on Deep Q Learning'''
    '''TODO: 
        1. add summary ops
        2. timing and logging
        3. model saving
        4. increment self.step
    '''

    def __init__(self, sess, logger, config, env):
        super(Agent, self).__init__(config)
        self.sess = sess
        self.logger = logger
        self.config = config
        params = DeepSenseParams(config)

        self.env = env
        self.history = History(logger, config)
        self.replay_memory = ReplayMemory(logger, config)

        with tf.variable_scope(STEPS):
            self.step_op = tf.Variable(0, trainable=False, name=STEP)
            self.step_input = tf.placeholder('int32', None, name=STEP_INPUT)
            self.step_assign_op = self.step_op.assign(self.step_input)

        self.build_dqn(params)

    def train(self):
        start_step = self.step_op.eval()

        num_episodes, self.update_count, ep_reward = 0, 0, 0.
        total_reward, self.total_loss, self.total_q = 0., 0., 0.
        max_avg_ep_reward = 0
        ep_rewards, actions = [], []

        screen, reward, action, terminal = self.env.new_random_episode(self.history)

        for self.step in range(start_step, self.max_step):
            if self.step == self.learn_start:
                num_episodes, self.update_count, ep_reward = 0, 0, 0.
                total_reward, self.total_loss, self.total_q = 0., 0., 0.
                ep_rewards, actions = [], []

            # 1. predict
            action = self.predict(self.history.get())
            # 2. act
            screen, reward, terminal = self.env.act(action, training=True)
            # 3. observe
            self.observe(screen, reward, action, terminal)

            if terminal:
                screen, reward, action, terminal = self.env.new_random_episode(self.history)
                num_episodes += 1
                ep_rewards.append(ep_reward)
                ep_reward = 0.

            else:
                ep_reward += reward

            actions.append(action)
            total_reward += reward

    
    def predict(self, s_t, test_ep=None):
        ep = test_ep or (self.ep_end +
            max(0., (self.ep_start - self.ep_end) \
            * (self.ep_end_t - max(0., self.step - self.learn_start)) / self.ep_end_t))

        if random.random() < ep:
            action = random.randrange(self.env.action_size)
        else:
            action = self.q.action.eval({self.s_t: [s_t]})[0]

        return action

    def observe(self, screen, reward, action, terminal):
        #clip reward in the range min to max
        reward = max(self.min_reward, min(self.max_reward, reward))
        
        self.history.add(screen)
        self.replay_memory.add(screen, reward, action, terminal)

        if self.step > self.learn_start:
            if self.step % self.train_frequency == 0:
                self.q_learning_mini_batch()

            if self.step % self.target_q_update_step == self.target_q_update_step - 1:
                self.update_target_network()

    def q_learning_mini_batch(self):
        if self.replay_memory.count >= self.replay_memory.history_length:
            s_t, action, reward, s_t_plus_1, terminal = self.replay_memory.sample()
            
            max_q_t_plus_1 = self.t_q.action.eval({self.t_s_t: s_t_plus_1})
            terminal = np.array(terminal) + 0.
            target_q = reward + (1 - terminal) * max_q_t_plus_1

            _, q_t, loss = self.sess.run([self.optimizer, self.q.values, self.loss], {
                self.target_q: target_q,
                self.action: action,
                self.s_t: s_t,
                self.learning_rate_step: self.step,
                })

            self.total_loss += loss
            self.total_q += q_t.mean()
            self.update_count += 1

    def build_dqn(self, params):
        with tf.variable_scope(PREDICTION):
            self.s_t = tf.placeholder(
                dtype=tf.float32,
                shape=[None, self.replay_memory.history_length, 
                            self.replay_memory.num_channels]
            )
        self.q = DeepSense(params, self.logger, self.sess, self.config, name=Q_NETWORK)
        self.q.build_model(self.s_t)

        with tf.variable_scope(TARGET):
            self.t_s_t = tf.placeholder(
                dtype=tf.float32,
                shape=[None, self.replay_memory.history_length, 
                            self.replay_memory.num_channels]
            )
        self.t_q = DeepSense(params, self.logger, self.sess, self.config, name=T_Q_NETWORK)
        self.t_q.build_model(self.t_s_t, train=False)

        with tf.variable_scope(UPDATE_TARGET_NETWORK):
            self.q_weights_placeholders = {}
            self.t_weights_assign_ops = {}

            for name in self.q.weights.keys():
                self.q_weights_placeholders[name] = tf.placeholder(
                            tf.float32,
                            self.q.weights[name].get_shape().as_list()
                        )
            for name in self.q.weights.keys():
                self.t_weights_assign_ops[name] = self.t_q.weights[name].assign(
                    self.q_weights_placeholders[name]
                )

        with tf.variable_scope(TRAINING):
            self.target_q = tf.placeholder(tf.float32, [None], name=TARGET_Q)
            self.action = tf.placeholder(tf.int64, [None], name=ACTION)
            
            action_one_hot = tf.one_hot(self.action, self.env.action_size, 
                                            1.0, 0.0, name=ACTION_ONE_HOT)
            q_acted = tf.reduce_sum(self.q.values * action_one_hot, 
                                        reduction_indices=1, name=Q_ACTED)
                                        
            with tf.variable_scope(LOSS):
                self.delta = self.target_q - q_acted

                self.global_step = tf.Variable(0, trainable=False)

                self.loss = tf.reduce_mean(clipped_error(self.delta), name=LOSS)
            with tf.variable_scope(OPTIMIZER):
                self.learning_rate_step = tf.placeholder(tf.int64, None, name=LEARNING_RATE_STEP)
                self.learning_rate_op = tf.maximum(self.learning_rate_minimum,
                    tf.train.exponential_decay(
                        self.learning_rate,
                        self.learning_rate_step,
                        self.learning_rate_decay_step,
                        self.learning_rate_decay,
                        staircase=True))

                self.optimizer = tf.train.RMSPropOptimizer(
                    self.learning_rate_op, momentum=0.95, epsilon=0.01).minimize(self.loss)

        # tf.initialize_all_variables().run()
        #initialize the q network and the target network with the same weights
        # self.update_target_network()

    def update_target_network(self):
        for name in self.q.weights.keys():
            self.t_weights_assign_ops[name].eval(
                {self.q_weights_placeholders[name]: self.q.weights[name].eval()}
            )
        