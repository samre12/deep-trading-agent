# Change Log

- provided support for `screen` and `vim` in docker image to run `tensorboard` within the container and to create and edit configuration files prior to training of the agent

- changed the implementation of *Dropout* from `tf.layers.dense` to `tf.nn.dropout` while using `tf.placeholder` for maintaining *dropout keep probabilities* for different prediction and training tasks of the *predication network*

- removed *BatchNormalization* from the *DeepSense* arhciteture - support to be included again in a future commit