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


## Documentation for initializeTensor function
# The network would be initialized using Tensorflow.
# This function sets up with learning rate of 0.05 and 10,000 episodes, along with number of bandits and actions.
# returns contextual_bandit, agent, weights, total_episode, total_reward, and init
def initializeTensor():
    tf.reset_default_graph()
    cBandit = contextual_bandit()
    myAgent = agent(lr=0.005, s_size=cBandit.num_bandits, a_size=cBandit.num_actions)
    weights = tf.trainable_variables()[0]  # The weights we will evaluate to look into the network.
    total_episodes = 10000  # Set total number of episodes to train agent on.
    total_reward = np.zeros([cBandit.num_bandits, cBandit.num_actions])  # Set scoreboard for bandits to 0.
    init = tf.global_variables_initializer()  # initializes global variables.
    return cBandit, myAgent, weights, total_episodes, total_reward, init


## Documentation for BanditTensor function
# We start with getting a state and decide to explore or exploit based on greedy strategies.
# Once we get the reward, using Contextual Bandit's pullArm function.
# Returns result in dataframe
def BanditTensor(cBandit, e, sess, myAgent, weights, total_reward, i, df):
    s = cBandit.getBandit()  # Get a state from the environment.
    r = np.random.rand(1)  # Sampling from normal distribution
    if r < e:  # Choose either a random action or one from our network.
        action = np.random.randint(cBandit.num_actions)  # explore
    else:
        action = sess.run(myAgent.chosen_action, feed_dict={myAgent.state_in: [s]})  # exploit
    reward = cBandit.pullArm(action)  # Get our reward for taking an action given a bandit.
    # Update the network.
    feed_dict = {myAgent.reward_holder: [reward], myAgent.action_holder: [action],
                 myAgent.state_in: [s]}  # feed-forwarding
    _, ww = sess.run([myAgent.update, weights], feed_dict=feed_dict)  # update
    # Update our running tally of scores.
    total_reward[s, action] += reward
    if i % 500 == 0:
        meanR = np.mean(total_reward, axis=1)
        df = df.append({'x': i, 'y': meanR[0]}, ignore_index=True)  # recording the rewards into the data
    return df
