import os
from os.path import join
import tensorflow as tf

from utils.constants import *
from utils.strings import *

from model.deepsenseparams import DeepSenseParams

class DeepSense:
    '''DeepSense Architecture for Q function approximation over Timeseries'''

    def __init__(self, deepsenseparams, logger, sess, config, name=DEEPSENSE):
        self.params = deepsenseparams
        self.logger = logger
        self.sess = sess
        self.__name__ = name

        self._weights = None

    @property
    def action(self):
        return self._action

    @property
    def avg_q_summary(self):
        return self._avg_q_summary

    @property
    def params(self):
        return self.params

    @property
    def name(self):
        return self.__name__

    @property
    def values(self):
        return self._values

    @property
    def weights(self):
        if self._weights is None:
            self._weights = {}
            variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 
                                            scope=self.__name__)
            for variable in variables:
                name = "/".join(variable.name.split('/')[1:])
                self._weights[name] = variable
        return self._weights

    '''
    def batch_norm_layer(self, inputs, train, name, reuse):
        return tf.layers.batch_normalization(
                                inputs=inputs,
                                training=train,
                                name=name,
                                reuse=reuse,
                                scale=True)
    '''

    def conv2d_layer(self, inputs, filter_size, kernel_size, padding, name, reuse, activation=None):
        return tf.layers.conv2d(
                        inputs=inputs,
                        filters=filter_size,
                        kernel_size=[1, kernel_size],
                        strides=(1, 1),
                        padding=padding,
                        activation=activation,
                        name=name,
                        reuse=reuse
                    )

    def dense_layer(self, inputs, num_units, name, reuse, activation=None):
        output = tf.layers.dense(
                        inputs=inputs,
                        units=num_units,
                        activation=activation,
                        name=name,
                        reuse=reuse
                    )
        return output

    def dropout_layer(self, inputs, keep_prob, name, is_conv=False):
        if is_conv:
            channels = tf.shape(inputs)[-1]
            return tf.nn.dropout(
                            inputs,
                            keep_prob=keep_prob,
                            name=name,
                            noise_shape=[
                                self.batch_size, 1, 1, channels
                            ]
                        )
        else:
            return tf.nn.dropout(
                        inputs,
                        keep_prob=keep_prob,
                        name=name
                    )        

    def build_model(self, state, reuse=False):
        inputs = state[0]
        trade_rem = state[1]
        with tf.variable_scope(self.__name__, reuse=reuse):
            with tf.name_scope(PHASE):
                self.phase = tf.placeholder(dtype=tf.bool)

            with tf.variable_scope(INPUT_PARAMS, reuse=reuse):
                self.batch_size = tf.shape(inputs)[0]

            inputs = tf.reshape(inputs, 
                        shape=[self.batch_size, 
                                self.params.split_size, 
                                self.params.window_size, 
                                self.params.num_channels])

            # self.debug1 = inputs
            with tf.variable_scope(CONV_LAYERS, reuse=reuse):
                window_size = self.params.window_size
                num_convs = len(self.params.filter_sizes)
                for i in range(0, num_convs):
                    with tf.variable_scope(CONV_LAYERS_.format(i + 1), reuse=reuse):
                        window_size = window_size - self.params.kernel_sizes[i] + 1
                        inputs = self.conv2d_layer(inputs, self.params.filter_sizes[i], 
                                                    self.params.kernel_sizes[i], 
                                                    self.params.padding,
                                                    CONV_.format(i + 1), 
                                                    reuse,
                                                    activation=tf.nn.relu)
                                                
                        inputs = self.dropout_layer(inputs,
                                                    self.params.dropoutkeepprobs.conv_keep_prob, 
                                                    DROPOUT_CONV_.format(i + 1),
                                                    is_conv=True)
                                    
            if self.params.padding == VALID:
                inputs = tf.reshape(inputs, 
                                    shape=[
                                            self.batch_size, 
                                            self.params.split_size, 
                                            window_size * self.params.filter_sizes[-1]
                                        ]
                            )
            else:
                inputs = tf.reshape(inputs, 
                                    shape=[
                                            self.batch_size, 
                                            self.params.split_size, 
                                            self.params.window_size * self.params.filter_sizes[-1]
                                        ]
                            )
            # self.debug2 = inputs

            gru_cells = []
            for i in range(0, self.params.gru_num_cells):
                cell = tf.contrib.rnn.GRUCell(
                    num_units=self.params.gru_cell_size,
                    reuse=reuse
                )        
                
                cell = tf.contrib.rnn.DropoutWrapper(
                    cell, 
                    output_keep_prob=self.params.dropoutkeepprobs.gru_keep_prob,
                    variational_recurrent=True,
                    dtype=tf.float32
                )
                
                gru_cells.append(cell)

            multicell = tf.contrib.rnn.MultiRNNCell(gru_cells)
            with tf.name_scope(DYNAMIC_UNROLLING):
                output, final_state = tf.nn.dynamic_rnn(
                    cell=multicell,
                    inputs=inputs,
                    dtype=tf.float32
                )
            output = tf.unstack(output, axis=1)[-1]
            # self.debug3 = output

            ''' 
            Append the information regarding the number of trades left in the episode
            '''
            trade_rem = tf.expand_dims(trade_rem, axis=1)
            output = tf.concat([output, trade_rem], axis=1)

            with tf.variable_scope(FULLY_CONNECTED, reuse=reuse):
                num_dense_layers = len(self.params.dense_layer_sizes)
                for i in range(0, num_dense_layers):
                    with tf.variable_scope(DENSE_LAYER_.format(i + 1), reuse=reuse):
                        output = self.dense_layer(output, self.params.dense_layer_sizes[i], 
                                                    DENSE_.format(i + 1), reuse, activation=tf.nn.relu)                    
                        
                        output = self.dropout_layer(output,
                                                    self.params.dropoutkeepprobs.dense_keep_prob,
                                                    DROPOUT_DENSE_.format(i + 1))
                        
                        
            self._values = self.dense_layer(output, self.params.num_actions, Q_VALUES, reuse)
            
            with tf.name_scope(AVG_Q_SUMMARY):
                avg_q = tf.reduce_mean(self._values, axis=0)
                self._avg_q_summary = []
                for idx in range(self.params.num_actions):
                    self._avg_q_summary.append(tf.summary.histogram('q/{}'.format(idx), avg_q[idx]))
                self._avg_q_summary = tf.summary.merge(self._avg_q_summary, name=AVG_Q_SUMMARY)
            
            self._action = tf.arg_max(self._values, dimension=1, name=ACTION)
