import random
import numpy as np

from utils.constants import *
from utils.strings import *

class Environment:
    '''Exchange Simulator for Bitcoin based upon per minute historical prices'''

    def __init__(self, logger, config):
        self.logger = logger
        self.action_size = 3
        self.history_length = config[HISTORY_LENGTH]
        self.horizon = config[HORIZON]

        self.historical_prices = None #fill this field with the historical data

        self.action_dict = {
            0: NEUTRAL,
            1: LONG,
            2: SHORT
        }

    def new_random_episode(self, history):
        self.liquid, self.borrow, self.long, self.short = 0., 0., 0, 0
        self.timesteps = 0
        
        self.current = random.randint(self.history_length, 
                                        len(self.historical_prices) - self.horizon)
        history.set_history(self.historical_prices[self.current - self.history_length:self.current])
        

    def act(self, action):
        price = self.historical_prices[self.current][0]

        if self.action_dict[action] is LONG:
            self.long = self.long + 1
            self.borrow = self.borrow + price
            
        else if self.action_dict[action] is SHORT:
            self.short = self.short + 1
            self.liquid = self.liquid + price
        
        self.timesteps = self.timesteps + 1
        if self.timesteps is not self.horizon:
            return self.historical_prices[self.current], 0, False
        else:
            reward = self.liquid - self.borrow + \
                        (self.long - self.short) * price
            return self.historical_prices[self.current], reward, True
