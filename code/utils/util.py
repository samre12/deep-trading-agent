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
    formatter = \
        logging.Formatter('%(asctime)s - %(pathname)s - Line No %(lineno)s - Level %(levelname)s - %(message)s')
    info_handler = logging.FileHandler(config[LOG_FILE])
    info_handler.setLevel(logging.INFO)
    info_handler.setFormatter(formatter)

    out_handler = logging.StreamHandler(sys.stdout)
    out_handler.setLevel(logging.DEBUG)
    out_handler.setFormatter(formatter)

    logger = logging.getLogger(name=DEEP_TRADING_AGENT)
    logger.setLevel(logging.INFO)
    logger.addHandler(info_handler)
    
    return logger

        
