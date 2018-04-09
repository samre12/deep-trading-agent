from utils.constants import *

class DropoutKeepProbs:
    '''Defines the keep probabilities for different dropout layers'''

    def __init__(self, conv_keep_prob=1.0, dense_keep_prob=1.0, gru_keep_prob=1.0):
        self.conv_keep_prob = conv_keep_prob
        self.dense_keep_prob = dense_keep_prob
        self.gru_keep_prob = gru_keep_prob

class DeepSenseParams:
    '''Defines the parameters for the DeepSense Q Network Architecture'''

    def __init__(self, config, dropoutkeeprobs = None):
        #Timeseries Parameters
        self.num_actions = config[NUM_ACTIONS]
        self.num_channels = config[NUM_CHANNELS]
        self.split_size = config[SPLIT_SIZE]
        self.window_size = config[WINDOW_SIZE]

        #Dropout Layer Parameters
        self._dropoutkeeprobs = dropoutkeeprobs

        #Convolution Layer Parameters
        self.filter_sizes = config[FILTER_SIZES]
        self.kernel_sizes = config[KERNEL_SIZES]
        self.padding = config[PADDING]

        #GRU Parameters
        self.gru_cell_size = config[GRU_CELL_SIZE]
        self.gru_num_cells = config[GRU_NUM_CELLS]

        #FullyConnected Network Parameters
        self.dense_layer_sizes = config[DENSE_LAYER_SIZES]

    @property
    def dropoutkeepprobs(self):
        return self._dropoutkeeprobs