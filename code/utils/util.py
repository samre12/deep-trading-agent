from ConfigParser import ConfigParser
import logging

from constants import *

def get_config_parser(filename):
    config = ConfigParser(allow_no_value=True)
    config.read(filename)
    return config

def get_logger(config):
    logger = logging.FileHandler(config.get(LOGGING, LOG_FILE))
    return logger

def print_and_log_message(message, logger):
    logging.info(message)
    logger.info(message)

def print_and_log_message_list(message_list, logger):
    for message in message_list:
        logging.info(message)
        logger.info(message)
