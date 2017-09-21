# Deep Trading Agent
Deep Reinforcement Learning based Trading Agent for Bitcoin using [DeepSense](https://arxiv.org/abs/1611.01942) Network for Q function approximation. <br>
The model is inspired by [Deep Q-Trading](http://cslt.riit.tsinghua.edu.cn/mediawiki/images/5/5f/Dtq.pdf)  where they solve a simplified trading problem for a single asset. For each trading day t, only one of the three actions: neutral(1), long(2) and short(3) are allowed and reward is obtained. Deep Q-Learning agent is trained to maximize the total accumulated rewards. <br>
I modify the existing model by using the *Deep Sense* architecture for *Q function* approximation.
