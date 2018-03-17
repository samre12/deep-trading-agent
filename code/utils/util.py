from ConfigParser import ConfigParser
import sys
import logging

from constants import *
from strings import DEEP_TRADING_AGENT

def get_config_parser(filename):
    config = ConfigParser(allow_no_value=True)
    config.read(filename)
    return config

def get_logger(config):
    formatter = logging.Formatter(logging.BASIC_FORMAT)   
    info_handler = logging.FileHandler(config[LOG_FILE])
    info_handler.setLevel(logging.DEBUG)
    info_handler.setFormatter(formatter)

    out_handler = logging.StreamHandler(sys.stdout)
    out_handler.setLevel(logging.INFO)
    out_handler.setFormatter(formatter)

    logger = logging.getLogger(name=DEEP_TRADING_AGENT)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(info_handler)
    logger.addHandler(out_handler)
    
    return logger
