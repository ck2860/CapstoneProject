import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import tensorflow.compat.v1 as tf
tf.logging.set_verbosity(tf.logging.ERROR)

## Documentation for Agent class
# This class sets up a contextual bandit agent via building a feed-forward in the network and training/updating the network.
# It would either return positive or negative reward.  Positive reinforcement is a reward for picking the most optimal arm; negative reinforcement is a penalty for not picking the best arm.


class agent():
    ## Create a new agent
    # The agent would have a learning rate, the size of a state, and the size of actions.
    def __init__(self, lr, s_size, a_size):
        self.state_in = tf.placeholder(tf.int32, shape=(1,))

        # Establishing the feed-forward part of the network in the next four lines.
        state_in_OH = tf.one_hot(self.state_in, s_size)
        output = tf.layers.dense(state_in_OH, a_size, tf.nn.sigmoid, use_bias=False, kernel_initializer=tf.ones_initializer())
        self.output = tf.reshape(output, [-1])
        self.chosen_action = tf.argmax(self.output, 0)  # The agent takes a state and produces an action.

        # In the next six lines, we establish the training the network.
        self.reward_holder = tf.placeholder(tf.float32, shape=(1,))  # feed the reward and chosen action into the network
        self.action_holder = tf.placeholder(tf.int32, shape=(1,))
        self.responsible_weight = tf.slice(self.output, self.action_holder, [1])
        self.loss = -(tf.log(self.responsible_weight)*self.reward_holder)  # to compute the loss and use it to update the network.
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.update = optimizer.minimize(self.loss)

