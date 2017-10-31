import time
from os.path import join

from progressbar import ProgressBar, Bar, Counter, Timer, \
            RotatingMarker, Percentage

import tensorflow as tf
import numpy as np

from argparse import ArgumentParser

from model.deepsense import DeepSense
from model.deepsenseparams import DeepSenseParams

from utils.config import get_config
from utils.constants import *
from utils.strings import *
from utils.util import *

def main(config_file_path):
    config_parser = get_config_parser(config_file_path)
    config = get_config(config_parser)
    logger = get_logger(config)

    with tf.Session() as sess:
        with tf.variable_scope(INPUT):
            inp = tf.placeholder(dtype=tf.float32, shape=[None, 100, 5])
        
        deepsense = DeepSense(DeepSenseParams(config), logger, sess)
        deepsense.build_model(inp)

        # variables_list1 = tf.get_collection(key=tf.GraphKeys.GLOBAL_VARIABLES, scope=DEEPSENSE)

        # deepsense2 = DeepSense(DeepSenseParams(config), logger, sess, name='deep')
        # deepsense2.build_model(inp)

        # variables_list2 = tf.get_collection(key=tf.GraphKeys.GLOBAL_VARIABLES, scope='deep')
        
        summary_writer = tf.summary.FileWriter(config[TENSORBOARD_LOG_DIR])
        summary_writer.add_graph(sess.graph)
        summary_writer.close()


if __name__ == "__main__":
    arg_parser = ArgumentParser(description='Deep Q Trading with DeepSense Architecture')
    arg_parser.add_argument('--config', dest='file_path',
                            help='Path for the configuration file')
    args = arg_parser.parse_args()
    main(vars(args)['file_path'])
