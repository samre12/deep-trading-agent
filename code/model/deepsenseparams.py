from utils.constants import *

class DeepSenseParams:
    '''Defines the parameters for the DeepSense Q Network Architecture'''

    def __init__(self, config):
        #Timeseries Parameters
        self.num_actions = config[NUM_ACTIONS]
        self.num_channels = config[NUM_CHANNELS]
        self.split_size = config[SPLIT_SIZE]
        self.window_size = config[WINDOW_SIZE]

        #Dropout Layer Parameters
        self.conv_keep_prob = config[CONV_KEEP_PROB]
        self.dense_keep_prob = config[DENSE_KEEP_PROB]
        self.gru_keep_prob = config[GRU_KEEP_PROB]

        #Convolution Layer Parameters
        self.filter_sizes = config[FILTER_SIZES]
        self.kernel_sizes = config[KERNEL_SIZES]

        #GRU Parameters
        self.gru_cell_size = config[GRU_CELL_SIZE]
        self.gru_num_cells = config[GRU_NUM_CELLS]

        #FullyConnected Network Parameters
        self.dense_layer_sizes = config[DENSE_LAYER_SIZES]