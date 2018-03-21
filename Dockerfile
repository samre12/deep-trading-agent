# Adapted from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/docker/Dockerfile
FROM ubuntu:14.04

LABEL maintainer="Craig Citro <craigcitro@google.com>"

# Pick up some TF dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        libfreetype6-dev \
        libpng12-dev \
        libzmq3-dev \
        pkg-config \
        python \
        python-dev \
        rsync \
        screen \
        software-properties-common \
        unzip \
        vim \
        wget \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

RUN pip --no-cache-dir install \
        Pillow \
        h5py \
        matplotlib \
        numpy \
        pandas \
        scipy \
        sklearn \
        tqdm 

#Install tensorflow version "1.1.0"
RUN pip --no-cache-dir install tensorflow==1.1.0
        
# --- DO NOT EDIT OR DELETE BETWEEN THE LINES --- #
# These lines will be edited automatically by parameterized_docker_build.sh. #
# COPY _PIP_FILE_ /
# RUN pip --no-cache-dir install /_PIP_FILE_
# RUN rm -f /_PIP_FILE_

# Create a sub-folder to contain all the code
RUN mkdir deep-trading-agent

# Add the entire repository content to a sub-folder
COPY . /deep-trading-agent/

# Download the latest dataset from the Bitcoincharts Archive
RUN mkdir /deep-trading-agent/data
RUN wget http://api.bitcoincharts.com/v1/csv/coinbaseUSD.csv.gz -P /deep-trading-agent/data/
RUN gunzip /deep-trading-agent/data/coinbaseUSD.csv.gz
RUN python2 /deep-trading-agent/code/preprocess.py --transactions /deep-trading-agent/data/coinbaseUSD.csv --dataset /deep-trading-agent/data/btc.csv 
RUN rm /deep-trading-agent/data/coinbaseUSD.csv

# Setup logging enviroment
RUN mkdir /deep-trading-agent/logs /deep-trading-agent/logs/saved_models /deep-trading-agent/logs/tensorboard
RUN touch /deep-trading-agent/logs/run.log

# TensorBoard
EXPOSE 6006
