import numpy as np
import pandas as pd
from talib.abstract import *

from utils.constants import *
from utils.strings import *
from utils.util import print_and_log_message, print_and_log_message_list

class Processor:
    '''Preprocessor for Bitcoin prices dataset as obtained by following the procedure 
    described in https://github.com/philipperemy/deep-learning-bitcoin'''

    def __init__(self, config, logger):
        self.dataset_path = config[DATASET_PATH]
        self.logger = logger
        self.history_length = config[HISTORY_LENGTH]
        self.horizon = config[HORIZON]

        self.preprocess()
        self.generate_attributes()

    @property
    def price_blocks(self):
        return self._price_blocks

    @property
    def timestamp_blocks(self):
        return self._timestamp_blocks
        
    def preprocess(self):
        data = pd.read_csv(self.dataset_path)
        message = 'Columns found in the dataset {}'.format(data.columns)
        print_and_log_message(message, self.logger)
        data = data.dropna()
        start_time_stamp = data['Timestamp'][0]
        timestamps = data['Timestamp'].apply(lambda x: (x - start_time_stamp) / 60)
        timestamps = timestamps - range(timestamps.shape[0])
        data.insert(0, 'blocks', timestamps)
        blocks = data.groupby('blocks')
        message = 'Number of blocks of continuous prices found are {}'.format(len(blocks))
        print_and_log_message(message, self.logger)
        
        self._data_blocks = []
        distinct_episodes = 0
        for name, indices in blocks.indices.items():
            if len(indices) > (self.history_length + self.horizon):
                self._data_blocks.append(blocks.get_group(name))
                distinct_episodes = distinct_episodes + (len(indices) - (self.history_length + self.horizon) + 1)

        data = None
        message_list = ['Number of usable blocks obtained from the dataset are {}'.format(len(self._data_blocks))]
        message_list.append('Number of distinct episodes for the current configuration are {}'.format(distinct_episodes))
        print_and_log_message_list(message_list, self.logger)

    def generate_attributes(self):
        self._price_blocks = []
        self._timestamp_blocks = []
        for data_block in self._data_blocks:
            weighted_prices = data_block['price_close'].values
            diff = np.diff(weighted_prices)
            diff = np.insert(diff, 0, 0)
            sma15 = SMA(data_block, timeperiod=15, price='price_close')
            sma30 = SMA(data_block, timeperiod=30, price='price_close')        
            
            price_block = np.column_stack((weighted_prices, diff, sma15, 
                                            weighted_prices - sma15, sma15 - sma30))
            price_block = pd.DataFrame(data=price_block)
            price_block.fillna(method='bfill', inplace=True)
                                
            self._price_blocks.append(price_block.as_matrix())
            self._timestamp_blocks.append(data_block['DateTime_UTC'].values)
        
        self._data_blocks = None #free memory
            