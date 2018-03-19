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
        software-properties-common \
        unzip \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

RUN pip --no-cache-dir install \
        Pillow \
        h5py \
        ipykernel \
        jupyter \
        matplotlib \
        numpy \
        pandas \
        scipy \
        sklearn \
        tqdm \
        && \
    python -m ipykernel.kernelspec

# --- DO NOT EDIT OR DELETE BETWEEN THE LINES --- #
# These lines will be edited automatically by parameterized_docker_build.sh. #
# COPY _PIP_FILE_ /
# RUN pip --no-cache-dir install /_PIP_FILE_
# RUN rm -f /_PIP_FILE_

# Install TensorFlow CPU version from central repo
RUN pip --no-cache-dir install \
    http://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.0.0-cp27-none-linux_x86_64.whl
# --- ~ DO NOT EDIT OR DELETE BETWEEN THE LINES --- #

# Add the entire repository content to a sub-folder
ADD . /deep-trading-agent

# Download the latest dataset from the Bitcoincharts Archive
RUN mkdir /deep-trading-agent/data
RUN wget http://api.bitcoincharts.com/v1/csv/coinbaseUSD.csv.gz -P /deep-trading-agent/data/
RUN gunzip /deep-trading-agent/data/coinbaseUSD.csv.gz

# Setup logging enviroment
RUN mkdir /deep-trading-agent/logs
RUN touch /deep-trading-agent/logs/run.log
RUN mkdir /deep-trading-agent/logs/saved_models
RUN mkdir /deep-trading-agent/logs/tensorboard

# TensorBoard
EXPOSE 6006
