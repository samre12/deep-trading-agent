import json
from ConfigParser import ConfigParser

from utils.constants import *

def get_config(config_parser):
    config = {}
    #Logging
    config[LOG_FILE] = config_parser.get(LOGGING, LOG_FILE)
    config[SAVE_DIR] = config_parser.get(LOGGING, SAVE_DIR)
    config[TENSORBOARD_LOG_DIR] = config_parser.get(LOGGING, TENSORBOARD_LOG_DIR)

    #Dataset Parameters
    config[BATCH_SIZE] = int(config_parser.get(DATASET, BATCH_SIZE))
    config[HISTORY_LENGTH] = int(config_parser.get(DATASET, HISTORY_LENGTH))
    config[MEMORY_SIZE] = int(config_parser.get(DATASET, MEMORY_SIZE))
    config[NUM_ACTIONS] = int(config_parser.get(DATASET, NUM_ACTIONS))
    config[NUM_CHANNELS] = int(config_parser.get(DATASET, NUM_CHANNELS))
    config[SPLIT_SIZE] = int(config_parser.get(DATASET, SPLIT_SIZE))
    config[WINDOW_SIZE] = int(config_parser.get(DATASET, WINDOW_SIZE))

    #Dropout Layer Parameters
    config[CONV_KEEP_PROB] = float(config_parser.get(DROPOUT, CONV_KEEP_PROB))
    config[DENSE_KEEP_PROB] = float(config_parser.get(DROPOUT, DENSE_KEEP_PROB))
    config[GRU_KEEP_PROB] = float(config_parser.get(DROPOUT, GRU_KEEP_PROB))

    #Convolution Layer Parameters
    config[FILTER_SIZES] = json.loads(config_parser.get(CONVOLUTION, FILTER_SIZES))
    config[KERNEL_SIZES] = json.loads(config_parser.get(CONVOLUTION, KERNEL_SIZES))

    #GRUCell Parameters
    config[GRU_CELL_SIZE] = int(config_parser.get(GRU, GRU_CELL_SIZE))
    config[GRU_NUM_CELLS] = int(config_parser.get(GRU, GRU_NUM_CELLS))

    #FullyConnected Layer Parameters
    config[DENSE_LAYER_SIZES] = json.loads(config_parser.get(DENSE, DENSE_LAYER_SIZES))

    return config
