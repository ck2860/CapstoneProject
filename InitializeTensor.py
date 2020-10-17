import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import tensorflow.compat.v1 as tf
from ContextualBandit import *
from ContextualBanditAgent import *

tf.logging.set_verbosity(tf.logging.ERROR)


## Documentation for Agent class
# This class sets up a contextual bandit agent via building a feed-forward in the network and training/updating the network. It would either return positive or negative reward.  Positive reinforcement is a reward for picking the most optimal arm; negative reinforcement is a penalty for not picking the best arm.

def initializeTensor():
    tf.reset_default_graph()
    cBandit = contextual_bandit()
    myAgent = agent(lr=0.005, s_size=cBandit.num_bandits, a_size=cBandit.num_actions)
    weights = tf.trainable_variables()[0]  # The weights we will evaluate to look into the network.
    total_episodes = 10000  # Set total number of episodes to train agent on.
    total_reward = np.zeros([cBandit.num_bandits, cBandit.num_actions])  # Set scoreboard for bandits to 0.
    init = tf.global_variables_initializer()
    return cBandit, myAgent, weights, total_episodes, total_reward, init


def BanditTensor(cBandit, e, sess, myAgent, weights):
    s = cBandit.getBandit()  # Get a state from the environment.
    # Choose either a random action or one from our network.
    r = np.random.rand(1)
    if r < e:
        action = np.random.randint(cBandit.num_actions)  # explore
    else:
        action = sess.run(myAgent.chosen_action, feed_dict={myAgent.state_in: [s]})  # exploit
    reward = cBandit.pullArm(action)  # Get our reward for taking an action given a bandit.
    # Update the network.
    feed_dict = {myAgent.reward_holder: [reward], myAgent.action_holder: [action],
                 myAgent.state_in: [s]}
    _, ww = sess.run([myAgent.update, weights], feed_dict=feed_dict)
    return s, action, reward
