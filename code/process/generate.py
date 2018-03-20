"""Taken from https://github.com/philipperemy/deep-learning-bitcoin/blob/master/data_manager.py """

import time
import datetime

import pandas as pd

def file_processor(data_file, target_file):
    print 'Reading bitcoin market data file from: {}.'.format(data_file)
    d = pd.read_table(data_file, sep=',', header=None, index_col=0, names=['price', 'volume'])    
    d.index = d.index.map(lambda ts: datetime.datetime.fromtimestamp(int(ts)))
    d.index.names = ['DateTime_UTC']
    p = pd.DataFrame(d['price'].resample('1Min').ohlc())
    p.columns = ['price_open', 'price_high', 'price_low', 'price_close']
    v = pd.DataFrame(d['volume'].resample('1Min').sum())
    v.columns = ['volume']
    p['volume'] = v['volume']
    unix_timestamps = p.index.map(lambda ts: int(time.mktime(ts.timetuple())))
    p.insert(0, 'Timestamp', unix_timestamps)

    p.to_csv(target_file, sep=',')
