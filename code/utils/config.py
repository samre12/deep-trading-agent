import json
from ConfigParser import ConfigParser

from utils.constants import *

def get_config(config_obj):
    config = {}
    #Dataset Parameters
    config[NUM_ACTIONS] = int(config_obj.get(DATASET, NUM_ACTIONS))
    config[SPLIT_SIZE] = int(config_obj.get(DATASET, SPLIT_SIZE))
    config[WINDOW_SIZE] = int(config_obj.get(DATASET, WINDOW_SIZE))

    #Dropout Layer Parameters
    config[CONV_KEEP_PROB] = float(config_obj.get(DROPOUT, CONV_KEEP_PROB))
    config[DENSE_KEEP_PROB] = float(config_obj.get(DROPOUT, DENSE_KEEP_PROB))
    config[GRU_KEEP_PROB] = float(config_obj.get(DROPOUT, GRU_KEEP_PROB))

    #Convolution Layer Parameters
    config[FILTER_SIZES] = json.loads(config_obj.get(CONVOLUTION, FILTER_SIZES))
    config[KERNEL_SIZES] = json.loads(config_obj.get(CONVOLUTION, KERNEL_SIZES))

    #GRUCell Parameters
    config[GRU_CELL_SIZE] = int(config_obj.get(GRU, GRU_CELL_SIZE))
    config[GRU_NUM_CELLS] = int(config_obj.get(GRU, GRU_NUM_CELLS))

    #FullyConnected Layer Parameters
    config[DENSE_LAYER_SIZES] = json.loads(config_obj.get(DENSE, DENSE_LAYER_SIZES))
