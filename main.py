import time
from os.path import join

import tensorflow as tf
import numpy as np
from argparse import ArgumentParser

import gym
import gym_cryptotrading

from agent.model.train import Agent

from agent.utils.config import get_config
from agent.utils.constants import *
from agent.utils.strings import *
from agent.utils.util import *

def main(config_file_path):
    config_parser = get_config_parser(config_file_path)
    config = get_config(config_parser)
    logger = get_logger(config)

    # Setting a lower debugging level for gym
    gym.logger.set_level(10)

    with tf.Session() as sess:
        # Setup environment
        env = gym.make('RealizedPnLEnv-v0')
        env.env.set_params(
            history_length=config[HISTORY_LENGTH],
            horizon=config[HORIZON],
            unit=5e-4
        )
        env.env.set_logger(logger)

        agent = Agent(sess, logger, config, env)
        agent.train()

        agent.summary_writer.close()

if __name__ == "__main__":
    arg_parser = ArgumentParser(description='Deep Q Trading with DeepSense Architecture')
    arg_parser.add_argument('--config', dest='file_path',
                            help='Path for the configuration file')
    args = arg_parser.parse_args()
    main(vars(args)['file_path'])
