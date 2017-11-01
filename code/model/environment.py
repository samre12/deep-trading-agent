import numpy as np

from utils.constants import *
from utils.strings import *

class Environment:
    '''Exchange Simulator for Bitcoin based upon per minute historical prices'''

    def __init__(self, logger, config):
        self.logger = logger
        self.action_size = 3
        self.actions = [
            NEUTRAL,
            LONG, 
            SHORT
        ]

    def new_random_episode(self, history):
        return None

    def act(self, action, training=True):
        return None