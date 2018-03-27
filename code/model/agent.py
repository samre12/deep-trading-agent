"""Code adapted from https://github.com/devsisters/DQN-tensorflow/tree/master/dqn/agent.py"""

import os
import sys
import time
import random
from tqdm import tqdm
import tensorflow as tf
import numpy as np

from model.baseagent import BaseAgent
from model.deepsense import DeepSense
from model.deepsenseparams import DeepSenseParams, DropoutKeepProbs
from model.environment import Environment
from model.history import History
from model.replay_memory import ReplayMemory
from model.util import clipped_error

from utils.constants import *
from utils.strings import *
                        
class Agent(BaseAgent):
    '''Deep Trading Agent based on Deep Q Learning'''
    '''TODO: 
        1. add `play` function to run tests in the simulated environment
    '''

    def __init__(self, sess, logger, config, env):
        super(Agent, self).__init__(config, logger)
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

    @property
    def summary_writer(self):
        return self._summary_writer

    def train(self):
        start_step = self.sess.run(self.step_op)

        num_episodes, self.update_count, ep_reward = 0, 0, 0.
        total_reward, self.total_loss, self.total_q = 0., 0., 0.
        max_avg_ep_reward = 0
        ep_rewards, actions = [], []

        trade_rem = self.env.new_random_episode(self.history, self.replay_memory)

        for self.step in tqdm(range(start_step, self.max_step), ncols=70, initial=start_step):
            if self.step == self.learn_start:
                num_episodes, self.update_count, ep_reward = 0, 0, 0.
                total_reward, self.total_loss, self.total_q = 0., 0., 0.
                ep_rewards, actions = [], []

            # 1. predict
            action = self.predict((self.history.history, trade_rem))
            # 2. act
            screen, reward, terminal, trade_rem = self.env.act(action)
            # 3. observe
            self.observe(screen, reward, action, terminal, trade_rem)

            if terminal:
                self.env.new_random_episode(self.history, self.replay_memory)
                num_episodes += 1
                ep_rewards.append(ep_reward)
                ep_reward = 0.

            else:
                ep_reward += reward

            actions.append(action)
            total_reward += reward
            
            if self.step >= self.learn_start:
                if self.step % self.test_step == self.test_step - 1:
                    avg_reward = total_reward / self.test_step
                    avg_loss = self.total_loss / self.update_count
                    avg_q = self.total_q / self.update_count

                    try:
                        max_ep_reward = np.max(ep_rewards)
                        min_ep_reward = np.min(ep_rewards)
                        avg_ep_reward = np.mean(ep_rewards)
                    except:
                        max_ep_reward, min_ep_reward, avg_ep_reward = 0, 0, 0

                    message = 'avg_r: %.4f, avg_l: %.6f, avg_q: %3.6f, avg_ep_r: %.4f, max_ep_r: %.4f, min_ep_r: %.4f, # game: %d' \
                        % (avg_reward, avg_loss, avg_q, avg_ep_reward, max_ep_reward, min_ep_reward, num_episodes)
                    self.logger.info(message)

                    if max_avg_ep_reward * 0.9 <= avg_ep_reward:
                        self.sess.run(
                            fetches=self.step_assign_op,
                            feed_dict={self.step_input: self.step + 1}
                        )
                        self.save_model(self.step + 1)

                        max_avg_ep_reward = max(max_avg_ep_reward, avg_ep_reward)

                    if self.step > 180:
                        self.inject_summary({
                            'average.reward': avg_reward,
                            'average.loss': avg_loss,
                            'average.q': avg_q,
                            'episode.max reward': max_ep_reward,
                            'episode.min reward': min_ep_reward,
                            'episode.avg reward': avg_ep_reward,
                            'episode.num of episodes': num_episodes,
                            'episode.rewards': ep_rewards,
                            'episode.actions': actions,
                            'training.learning_rate': self.sess.run(
                                fetches=self.learning_rate_op,
                                feed_dict={self.learning_rate_step: self.step}
                            )
                        }, self.step)

                    num_episodes = 0
                    total_reward = 0.
                    self.total_loss = 0.
                    self.total_q = 0.
                    self.update_count = 0
                    ep_reward = 0.
                    ep_rewards = []
                    actions = []
    
    def predict(self, state, test_ep=None):
        s_t = state[0]
        trade_rem_t = state[1]
        ep = test_ep or (self.ep_end +
            max(0., (self.ep_start - self.ep_end) \
            * (self.ep_end_t - max(0., self.step - self.learn_start)) / self.ep_end_t))

        if random.random() < ep:
            action = random.randrange(self.config[NUM_ACTIONS])
        else:
            action = self.sess.run(
                fetches=self.q.action,
                feed_dict={
                    self.q.phase: 0,  
                    self.s_t: [s_t], 
                    self.trade_rem_t: [trade_rem_t],
                    self.q_conv_keep_prob: 1.0,
                    self.q_dense_keep_prob: 1.0,
                    self.q_gru_keep_prob: 1.0
                }
            )[0]

        return action

    def observe(self, screen, reward, action, terminal, trade_rem):
        #clip reward in the range min to max
        reward = max(self.min_reward, min(self.max_reward, reward))
        
        self.history.add(screen)
        self.replay_memory.add(screen, reward, action, terminal, trade_rem)

        if self.step > self.learn_start:
            if self.step % self.train_frequency == 0:
                self.q_learning_mini_batch()

            if self.step % self.target_q_update_step == self.target_q_update_step - 1:
                self.update_target_network()

    def q_learning_mini_batch(self):
        if self.replay_memory.count >= self.replay_memory.history_length:
            state_t, action, reward, state_t_plus_1, terminal = self.replay_memory.sample
            s_t, trade_rem_t = state_t[0], state_t[1]
            s_t_plus_1, trade_rem_t_plus_1 = state_t_plus_1[0], state_t_plus_1[1]
            
            q_t_plus_1 = self.sess.run(
                fetches=self.t_q.values,
                feed_dict={
                    self.t_q.phase: 0, 
                    self.t_s_t: s_t_plus_1, 
                    self.t_trade_rem_t: trade_rem_t_plus_1
                }
            )

            max_q_t_plus_1 = np.max(q_t_plus_1, axis=1)

            terminal = np.array(terminal) + 0.
            target_q = reward + (1 - terminal) * max_q_t_plus_1

            _, q_t, loss, avg_q_summary = self.sess.run([self.optimizer, self.q.values, self.loss, self.q.avg_q_summary], {
                self.q.phase: 1,
                self.target_q: target_q,
                self.action: action,
                self.s_t: s_t,
                self.trade_rem_t: trade_rem_t,
                self.q_conv_keep_prob: self.config[CONV_KEEP_PROB],
                self.q_dense_keep_prob: self.config[DENSE_KEEP_PROB],
                self.q_gru_keep_prob: self.config[GRU_KEEP_PROB],
                self.learning_rate_step: self.step
            })

            self.summary_writer.add_summary(avg_q_summary, self.step)
            self.total_loss += loss
            self.total_q += q_t.mean()
            self.update_count += 1

    def build_dqn(self, params):
        with tf.variable_scope(PREDICTION):
            self.s_t = tf.placeholder(
                dtype=tf.float32,
                shape=[None, self.replay_memory.history_length, 
                            self.replay_memory.num_channels],
                name=HISTORICAL_PRICES
            )
            self.trade_rem_t = tf.placeholder(
                dtype=tf.float32,
                shape=[None,],
                name=TRADE_REM
            )
            
            with tf.variable_scope(DROPOUT_KEEP_PROBS):
                self.q_conv_keep_prob = tf.placeholder(tf.float32)
                self.q_dense_keep_prob = tf.placeholder(tf.float32)
                self.q_gru_keep_prob = tf.placeholder(tf.float32)

        params.dropoutkeepprobs = DropoutKeepProbs(
                    self.q_conv_keep_prob,
                    self.q_dense_keep_prob,
                    self.q_gru_keep_prob
                )
        self.q = DeepSense(params, self.logger, self.sess, self.config, name=Q_NETWORK)
        self.q.build_model((self.s_t, self.trade_rem_t))

        with tf.variable_scope(TARGET):
            self.t_s_t = tf.placeholder(
                dtype=tf.float32,
                shape=[None, self.replay_memory.history_length, 
                            self.replay_memory.num_channels],
                name=HISTORICAL_PRICES
            )
            self.t_trade_rem_t = tf.placeholder(
                dtype=tf.float32,
                shape=[None,],
                name=TRADE_REM
            )

        params.dropoutkeepprobs = DropoutKeepProbs()
        self.t_q = DeepSense(params, self.logger, self.sess, self.config, name=T_Q_NETWORK)
        self.t_q.build_model((self.t_s_t, self.t_trade_rem_t))

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
            
            action_one_hot = tf.one_hot(self.action, self.config[NUM_ACTIONS], 
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

        with tf.variable_scope(SUMMARY):
            scalar_summary_tags = ['average.reward', 'average.loss', 'average.q', \
                'episode.max reward', 'episode.min reward', 'episode.avg reward', \
                'episode.num of episodes', 'training.learning_rate']            

            self.summary_placeholders = {}
            self.summary_ops = {}

            for tag in scalar_summary_tags:
                self.summary_placeholders[tag] = \
                    tf.placeholder('float32', None, name=tag.replace(' ', '_'))
                self.summary_ops[tag] = \
                    tf.summary.scalar(
                        name="{}-{}".format(self.env_name, tag.replace(' ', '_')),
                        tensor=self.summary_placeholders[tag]
                    )

            histogram_summary_tags = ['episode.rewards', 'episode.actions']
            for tag in histogram_summary_tags:
                self.summary_placeholders[tag] = \
                    tf.placeholder('float32', None, name=tag)
                self.summary_ops[tag] = \
                    tf.summary.histogram(
                        tag,
                        self.summary_placeholders[tag]
                    )

        self.sess.run(tf.local_variables_initializer())
        self.sess.run(tf.global_variables_initializer())
        self._saver = tf.train.Saver(self.q.weights.values() + [self.step_op], max_to_keep=30)
        
        self.load_model()
        self.update_target_network()

        self._summary_writer = tf.summary.FileWriter(self.config[TENSORBOARD_LOG_DIR])
        self._summary_writer.add_graph(self.sess.graph)

    def update_target_network(self):
        for name in self.q.weights.keys():
            self.sess.run(
                fetches=self.t_weights_assign_ops[name],
                feed_dict=
                {self.q_weights_placeholders[name]: self.sess.run(
                    fetches=self.q.weights[name]
                )}
            )
    
    def inject_summary(self, tag_dict, step):
        summary_str_lists = self.sess.run([self.summary_ops[tag] for tag in tag_dict.keys()], {
            self.summary_placeholders[tag]: value for tag, value in tag_dict.items()
        })
        for summary_str in summary_str_lists:
            self.summary_writer.add_summary(summary_str, self.step)
        