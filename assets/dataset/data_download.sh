#!/usr/bin/env bash
#Taken from "https://github.com/philipperemy/deep-learning-bitcoin/blob/master/data_download.sh"
wget http://api.bitcoincharts.com/v1/csv/coinbaseUSD.csv.gz -P ./
gunzip ./coinbaseUSD.csv.gz