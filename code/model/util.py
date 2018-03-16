"""Code adapted from https://github.com/devsisters/DQN-tensorflow/tree/master/dqn/utils.py"""

try:
    import cPickle as pickle
except:
    import pickle

import tensorflow as tf

from utils.strings import *

def clipped_error(x):
    # Huber loss
    try:
        return tf.select(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)
    except:
        return tf.where(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)

def save_npy(obj, path, logger):
    np.save(path, obj)
    message = "  [*] saved at {}".format(path)
    logger.info(message)

def load_npy(path, logger):
    obj = np.load(path)
    message = "  [*] loaded from {}".format(path)
    logger.info(message)
    return obj