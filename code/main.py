import time
from os.path import join

from progressbar import ProgressBar, Bar, Counter, Timer, \
            RotatingMarker, Percentage

import tensorflow as tf
import numpy as np

from argparse import ArgumentParser

from model.deepsense import DeepSense
from model.deepsenseparams import DeepSenseParams
from model.util import load_pre_trained_model

from utils.config import get_config
from utils.constants import *
from utils.strings import *
from utils.util import *

def main(config_file_path):
    config_parser = get_config_parser(config_file_path)
    config = get_config(config_parser)
    logger = get_logger(config)

    with tf.Session() as sess:
        latest_checkpoint = tf.train.latest_checkpoint(config[SAVE_DIR])
        if latest_checkpoint is not None:
            '''This will load a pretrained model into the current session graph'''
            message = 'Found lateset checkpoint at : {}'.format(latestcheckpoint)
            print_and_log_message(message, logger)

            load_pre_trained_model(sess, latestcheckpoint, logger) 

        else:
            with tf.variable_scope(INPUT):
                inp = tf.placeholder(dtype=tf.float32, shape=[None, 100, 5])
            
            deepsense = DeepSense(DeepSenseParams(config), logger)
            deepsense.build_model(inp)

        q_values = tf.get_collection(Q_VALUES)[0]
        
        summary_writer = tf.summary.FileWriter(config[TENSORBOARD_LOG_DIR])
        summary_writer.add_graph(sess.graph)

        saver = tf.train.Saver(pad_step_number=True)

        summary_writer.close()


if __name__ == "__main__":
    arg_parser = ArgumentParser(description='Deep Q Trading with DeepSense Architecture')
    arg_parser.add_argument('--config', dest='file_path',
                            help='Path for the configuration file')
    args = arg_parser.parse_args()
    main(vars(args)['file_path'])
