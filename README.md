# Deep Trading Agent
[![license](https://img.shields.io/packagist/l/doctrine/orm.svg)](https://github.com/samre12/deep-trading-agent/blob/master/LICENSE)
[![dep1](https://img.shields.io/badge/implementation-tensorflow-orange.svg)](https://www.tensorflow.org/)
[![dep2](https://img.shields.io/badge/python-2.7-red.svg)](https://www.python.org/download/releases/2.7/)
[![dep3](https://img.shields.io/badge/status-in%20progress-green.svg)](https://github.com/samre12/deep-trading-agent/)<br>
Deep Reinforcement Learning based Trading Agent for Bitcoin using [DeepSense](https://arxiv.org/abs/1611.01942) Network for Q function approximation. <br><br>
![model](assets/schema/CompleteSchema.png)
<br>
For complete details of the dataset, preprocessing, network architecture and implementation, refer to the [Wiki](https://github.com/samre12/deep-trading-agent/wiki) of this repository.

## Requirements
- Python 2.7
- [Tensorflow](https://www.tensorflow.org/)
- [TA-Lib](https://mrjbq7.github.io/ta-lib/) (for processing Bitcoin Price Series)
- [Pandas](https://pandas.pydata.org) (for pre-processing Bitcoin Price Series)
- [tqdm](https://pypi.python.org/pypi/tqdm) (for displaying progress of training)<br>

To setup a ubuntu virtual machine with all the dependencies to run the code, refer to `assets/vm`.

## Support
Please give a :star: to this repository to support the project :smile:.

## ToDo
- [ ] Fix the model to ensure convergence of state action function to positive values
- [ ] Add Docket support for a fast and easy start with the project

## Trading Model
is inspired by [Deep Q-Trading](http://cslt.riit.tsinghua.edu.cn/mediawiki/images/5/5f/Dtq.pdf) where they solve a simplified trading problem for a single asset. <br>
For each trading unit, only one of the three actions: neutral(1), long(2) and short(3) are allowed and a reward is obtained depending upon the current position of agent. Deep Q-Learning agent is trained to maximize the total accumulated rewards. <br>
Current Deep Q-Trading model is modified by using the *Deep Sense* architecture for *Q function* approximation.

## Dataset
Per minute Bitcoin series is obtained by modifying the procedure mentioned in [this](https://github.com/philipperemy/deep-learning-bitcoin) repository. Transactions in the *Coinbase* exchange are sampled to generate the Bitcoin price series. <br>
Refer to `assets/dataset` to download the dataset.

### Preprocessing
**Basic Preprocessing**<br>
Completely ignore missing values and remove them from the dataset and accumulate blocks of continuous values using the timestamps of the prices.<br>
All the accumulated blocks with number of timestamps lesser than the combined *history length* of the state and *horizon* of the agent are then filtered out since they cannot be used for training of the agent.<br>
In the current implementation, past 3 hours (180 minutes) of per minute Bitcoin prices are used to generate the representation of the current state of the agent.<br>
With the existing dataset (at the time of writing), following are the logs generated while preprocessing the dataset:
```
INFO:root:Number of blocks of continuous prices found are 58863
INFO:root:Number of usable blocks obtained from the dataset are 887
INFO:root:Number of distinct episodes for the current configuration are 558471
```

**Advanced Preprocessing**<br>
Process missing values and concatenate smaller blocks to increase the sizes of continuous price blocks<br>
*(To be implemented)*

## Implementation
Tensorflow "1.1.0" version is used for the implementation of the **Deep Sense** network.<br>
### Deep Sense
Implementation is adapted from [this](https://github.com/yscacaca/DeepSense) Github repository with a few simplifications in the network architecture to incorporate learning over a single time series of the Bitcoin data.

### Deep Q Trading
Implementation and preprocessing is inspired from this [Medium post](https://hackernoon.com/the-self-learning-quant-d3329fcc9915). The actual implementation of the Deep Q Network is adapted from [DQN-tensorflow](https://github.com/devsisters/DQN-tensorflow).
