from __future__ import print_function
import os
import time
import random
import tensorflow as tf
import numpy as np

from model.deepsense import DeepSense
from model.deepsenseparams import DeepSenseParams

from utils.constants import *
from utils.strings import *
from utils.util import print_and_log_message, print_and_log_message_list

class Agent:
    '''Deep Trading Agent based on Deep Q Learning'''


