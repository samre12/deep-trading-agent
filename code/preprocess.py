import time

from argparse import ArgumentParser

from process.generate import file_processor

from utils.constants import *
from utils.strings import *

def main(args):
    file_processor(vars(args)['transactions'], vars(args)['dataset']) 

if __name__ == "__main__":
    arg_parser = ArgumentParser(description='Deep Q Trading with DeepSense Architecture')
    arg_parser.add_argument('--transactions', dest='transactions',
                            help='Path for the input transactions at the exchange')
    arg_parser.add_argument('--dataset', dest='dataset',
                            help='Path for the sampled dataset')
    
    args = arg_parser.parse_args()
    main(args)
    