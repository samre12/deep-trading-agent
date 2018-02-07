import random
import numpy as np

from utils.constants import *
from utils.strings import *
from utils.util import print_and_log_message_list

class Environment:
    '''Exchange Simulator for Bitcoin based upon per minute historical prices'''

    def __init__(self, logger, config, price_blocks, timestamp_blocks):
        self.logger = logger
        self.episode_number = 0
        self.history_length = config[HISTORY_LENGTH]
        self.horizon = config[HORIZON]

        self.price_blocks = price_blocks
        self.timestamp_blocks = timestamp_blocks

        self.action_dict = {
            0: NEUTRAL,
            1: LONG,
            2: SHORT
        }

        self.unit = 5e-5 #units of Bitcoin traded each time

    def new_random_episode(self, history, replay_memory):
        '''
        TODO: In the current setting, the selection of an episode does not follow pure uniform process. 
        Need to index every episode and then generate a random index rather than going on multiple levels
        of selection.
        '''
        message_list = []
        self.episode_number = self.episode_number + 1
        message_list.append("Starting a new episode numbered {}".format(self.episode_number))
        self.liquid, self.borrow, self.long, self.short = 0., 0., 0, 0
        self.timesteps = 0
        
        block_index = random.randint(0, len(self.price_blocks) - 1)
        message_list.append("Block index selected for episode number {} is {}".format(
            self.episode_number, block_index))
        self.historical_prices = self.price_blocks[block_index]

        self.current = random.randint(self.history_length,  
                                        len(self.historical_prices) - self.horizon)
        message_list.append("Starting index and timestamp point selected for episode number {} is {}:==:{}".format(
            self.episode_number, self.current, self.timestamp_blocks[block_index][self.current]
        ))
        
        #Set history and replay memory
        for state in self.historical_prices[self.current - self.history_length:self.current]:
            history.add(state)
            replay_memory.add(state, 0.0, 0, False)
            
        print_and_log_message_list(message_list, self.logger)

    def act(self, action):
        state = self.historical_prices[self.current]
        price = state[0]

        if self.action_dict[action] is LONG:
            self.long = self.long + 1
            self.borrow = self.borrow + (price * self.unit)
            
        elif self.action_dict[action] is SHORT:
            self.short = self.short + 1
            self.liquid = self.liquid + (price * self.unit)
        
        self.timesteps = self.timesteps + 1
        if self.timesteps is not self.horizon:
            self.current = self.current + 1
            return state, 0, False
        else:
            reward = self.liquid - self.borrow + \
                        (self.long - self.short) * price * self.unit
            return state, reward, True
