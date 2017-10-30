import tensorflow as tf

from utils.util import print_and_log_message, print_and_log_message_list
from utils.constants import *
from utils.strings import *

from model.deepsenseparameters import DeepSenseParams

class DeepSense:
    '''DeepSense Architecture for Q function approximation over Timeseries'''

    def __init__(deepsenseparams, logger):
        self.params = deepsenseparams
        self.logger = logger

    def batch_norm_layer(self, inputs, train, name, reuse):
        return tf.layers.batch_normalization(
                                inputs=inputs,
                                trainable=train,
                                name=name,
                                reuse=reuse,
                                scale=True)

    def conv2d_layer(self, inputs, filer_size, kernel_size, name, reuse):
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

    def dense_layer(self, inputs, num_units, name, resue, activation=None):
        return tf.layers.dense(
                        inputs=inputs,
                        units=num_units,
                        activation=tf.nn.relu,
                        name=name,
                        resue=reuse
                    )

    def dropout_layer(self, inputs, train, keep_prob, name):
        return tf.layers.dropout(
                        inputs=inputs,
                        rate=keep_prob,
                        training=train,
                        name=name,
                        noise_shape=[
                            self.batch_size, 1, 1, self.channels
                        ]
                    )

    def build_model(self, inputs, train=True, reuse=False, name="DeepSense"):
        self.batch_size = inputs.get_shape().as_list()[0]
        self.channels = inputs.get_shape().as_list()[-1]

        with tf.varialbe_scope("DeepSense", reuse=reuse):
            inputs = tf.reshape(inputs, shape=[batch_size, split_size, window_size, channels])

            num_convs = len(self.params.filter_sizes)
            for i in range(0, num_convs):
                    if i > 0:
                        inputs = self.dropout_layer(inputs, train, 
                                                    self.params.conv_keep_prob, 
                                                    DROPOUT_CONV_.format(i))
                    inputs = self.conv2d_layer(inputs, self.params.filer_size[i], 
                                                self.params.kernel_size[i], 
                                                CONV_.format(i + 1), 
                                                reuse)
                    inputs = self.batch_norm_layer(inputs, train, 
                                                    BATCH_NORM_.format(i + 1), reuse)
                    inputs = tf.nn.relu(inputs)
            
            inputs = tf.reshape(inputs, shape=[self.batch_size, split_size, -1])

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
            output, final_state = tf.nn.dynamic_rnn(
                cell=multicell,
                inputs=inputs,
                dtype=tf.float32
            )
            output = tf.unstack(output, axis=1)[-1]

            num_dense_layers = len(self.params.dense_layer_sizes)
            for i in range(num_dense_layers):
                if i > 0:
                    output = self.dropout_layer(output, train, 
                                                self.params.dense_keep_prob,
                                                DROPOUT_DENSE_.format(i))
                output = self.dense_layer(output, size, name, reuse, tf.nn.relu)

            q_values = self.dense_layer(output, self.num_actions, Q_VALUES, reuse)
            return q_values
        
