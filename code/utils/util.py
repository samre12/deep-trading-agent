from ConfigParser import ConfigParser
import logging
import tensorflow as tf

from constants import *
from strings import DEEP_TRADING_AGENT

def get_config_parser(filename):
    config = ConfigParser(allow_no_value=True)
    config.read(filename)
    return config

def get_logger(config):
    logging.basicConfig(level=logging.DEBUG)
    formatter = \
        logging.Formatter('%(asctime)s - %(pathname)s - Line No %(lineno)s - Level %(levelname)s - %(message)s')
    info_handler = logging.FileHandler(config[LOG_FILE])
    info_handler.setLevel(logging.INFO)
    info_handler.setFormatter(formatter)

    logger = logging.getLogger(name=DEEP_TRADING_AGENT)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(info_handler)
    logger.propagate = False
    return logger

def print_and_log_message(message, logger):
    logging.info(message)
    logger.info(message)

def print_and_log_message_list(message_list, logger):
    for message in message_list:
        logging.info(message)
        logger.info(message)

def clipped_error(x):
    # Huber loss
    try:
        return tf.select(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)
    except:
        return tf.where(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)
