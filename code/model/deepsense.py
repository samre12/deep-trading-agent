import tensorflow as tf

from utils.util import print_and_log_message, print_and_log_message_list
from utils.constants import *
from utils.strings import *

from model.deepsenseparams import DeepSenseParams

class DeepSense:
    '''DeepSense Architecture for Q function approximation over Timeseries'''

    def __init__(self, deepsenseparams, logger):
        self.params = deepsenseparams
        self.logger = logger

    def batch_norm_layer(self, inputs, train, name, reuse):
        return tf.layers.batch_normalization(
                                inputs=inputs,
                                trainable=train,
                                name=name,
                                reuse=reuse,
                                scale=True)

    def conv2d_layer(self, inputs, filter_size, kernel_size, name, reuse):
        return tf.layers.conv2d(
                        inputs=inputs,
                        filters=filter_size,
                        kernel_size=[1, kernel_size],
                        strides=(1, 1),
                        padding='valid',
                        activation=None,
                        name=name,
                        reuse=reuse
                    )

    def dense_layer(self, inputs, num_units, name, reuse, activation=None):
        return tf.layers.dense(
                        inputs=inputs,
                        units=num_units,
                        activation=activation,
                        name=name,
                        reuse=reuse
                    )

    def dropout_conv_layer(self, inputs, train, keep_prob, name):
        channels = tf.shape(inputs)[-1]
        return tf.layers.dropout(
                        inputs=inputs,
                        rate=keep_prob,
                        training=train,
                        name=name,
                        noise_shape=[
                            self.batch_size, 1, 1, channels
                        ]
                    )

    def dropout_dense_layer(self, inputs, train, keep_prob, name):
        return tf.layers.dropout(
                        inputs=inputs,
                        rate=keep_prob,
                        training=train,
                        name=name
                    )        

    def build_model(self, inputs, train=True, reuse=False, name=DEEPSENSE):
        with tf.variable_scope(INPUT_PARAMS, reuse=reuse):
            self.batch_size = tf.shape(inputs)[0]
        
        with tf.variable_scope(name, reuse=reuse):
            inputs = tf.reshape(inputs, 
                        shape=[self.batch_size, 
                                self.params.split_size, 
                                self.params.window_size, 
                                self.params.num_channels])

            with tf.variable_scope(CONV_LAYERS, reuse=reuse):
                window_size = self.params.window_size
                num_convs = len(self.params.filter_sizes)
                for i in range(0, num_convs):
                    with tf.variable_scope(CONV_LAYERS_.format(i + 1), reuse=reuse):
                        window_size = window_size - self.params.kernel_sizes[i] + 1
                        inputs = self.conv2d_layer(inputs, self.params.filter_sizes[i], 
                                                    self.params.kernel_sizes[i], 
                                                    CONV_.format(i + 1), 
                                                    reuse)
                        inputs = self.batch_norm_layer(inputs, train, 
                                                        BATCH_NORM_.format(i + 1), reuse)
                        inputs = tf.nn.relu(inputs)
                        if i < num_convs - 1:
                            inputs = self.dropout_conv_layer(inputs, train, 
                                                        self.params.conv_keep_prob, 
                                                        DROPOUT_CONV_.format(i + 1))
            
            input_shape = tf.shape(inputs)
            inputs = tf.reshape(inputs, shape=[self.batch_size, self.params.split_size, 
                                                window_size * self.params.filter_sizes[-1]])

            gru_cells = []
            for i in range(0, self.params.gru_num_cells):
                cell = tf.contrib.rnn.GRUCell(
                    num_units=self.params.gru_cell_size,
                    reuse=reuse
                )
                if train:
                    cell = tf.contrib.rnn.DropoutWrapper(
                        cell, output_keep_prob=self.params.gru_keep_prob
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

            with tf.variable_scope(FULLY_CONNECTED, reuse=reuse):
                num_dense_layers = len(self.params.dense_layer_sizes)
                for i in range(0, num_dense_layers):
                    with tf.variable_scope(DENSE_LAYER_.format(i + 1), reuse=reuse):
                        output = self.dense_layer(output, self.params.dense_layer_sizes[i], 
                                                    DENSE_.format(i + 1), reuse, tf.nn.relu)
                        if i < num_dense_layers - 1:
                            output = self.dropout_dense_layer(output, train, 
                                                        self.params.dense_keep_prob,
                                                        DROPOUT_DENSE_.format(i + 1))

            q_values = self.dense_layer(output, self.params.num_actions, Q_VALUES, reuse)
            tf.add_to_collection(Q_VALUES, q_values)
