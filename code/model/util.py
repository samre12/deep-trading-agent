import tensorflow as tf

from utils.util import print_and_log_message_list
from utils.strings import *


def load_pre_trained_model(sess, model_load_path, logger):
    '''Method to load the pre-trained model from the disk'''
    message_list = []
    try:
        message_list.append("Loading model from " + model_load_path)
        saver = tf.train.import_meta_graph(model_load_path + '.meta')
        saver.restore(sess, save_path=model_load_path)

        message_list.append("Model successfully loaded")

    except Exception:
        message_list.append(PRE_TRAINED_MODEL_NOT_LOADED)
    
    finally:
        print_and_log_message_list(message_list, logger)

def load_model_weights(sess, weights_load_path, logger):
    '''Method to load weights of the learned variables for inference'''
    message_list = []
    try:
        message_list.append("Loading model from " + weights_load_path)
        weights_loader = tf.train.Saver(
            name=WEIGHTS_LOADER
        )
        weights_loader.restore(sess, save_path=weights_load_path)
        message_list.append("Model weights successfully loaded")
    
    except Exception:
        message_list.append(MODEL_WEIGHTS_NOT_LOADED)
    
    finally:
        print_and_log_message_list(message_list, logger)
        
        