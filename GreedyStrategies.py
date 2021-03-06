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
from InitializeTensor import *


## @package GreedyStrategies
## Documentation for Greedy Strategies class
# this class has decreasing-epsilon, greedy-epsilon, hybrid#1, hybrid#2, hybrid#3, hybrid#4, and hybrid#5 functions.
# ContextualBandit and ContextualBanditAgent classes are used here. Please go to see Class tab for more details.

class greedyStrategies():
    def __init__(self, n):
        self.numberOfTrials = n

    ## Documentation for Decreasing Epsilon function
    # this would return the rewards of Decreasing Epsilon
    ##@param a1: random seed
    ##@param df1a: initialized dataframe
    def decreasingEpsilon(self, a1, df1a):
        for a in range(self.numberOfTrials):
            cBandit, myAgent, weights, total_episodes, total_reward, init = initializeTensor()
            e = 1  # start with highly explorative (100% explore, 0% exploit)
            value = 0.0001  # the value that we will decrement the epsilon
            # Launch the tensorflow graph
            with tf.Session() as sess:
                sess.run(init)
                i = 0
                np.random.seed(a1)
                while i < total_episodes:
                    df1a = BanditTensor(cBandit, e, sess, myAgent, weights, total_reward, i, df1a)
                    e -= value  # in the end it would be highly exploitative
                    i += 1
                a1 += 1
                print("Trial#", a + 1, ": Epsilon-Decreasing is done! ")
        return df1a

    ## Documentation for Epsilon-Greedy function
    # this would return the rewards of Epsilon-Greedy
    ##@param b: random seed
    ##@param df2a: initialized dataframe
    def epsilonGreedy(self, b, df2a):
        for a in range(self.numberOfTrials):
            cBandit, myAgent, weights, total_episodes, total_reward, init = initializeTensor()
            e = 0.1  # this is the probability of the agent to explore. 10% to explore and 90% of exploit
            # Launch the tensorflow graph
            with tf.Session() as sess:
                sess.run(init)
                i = 0
                np.random.seed(b)
                while i < total_episodes:
                    df2a = BanditTensor(cBandit, e, sess, myAgent, weights, total_reward, i, df2a)
                    i += 1
                b += 1
                print("Trial#", a + 1, ": Epsilon-Greedy is done!")
        return df2a

    ## Documentation for Hybrid #1 function
    # this would return the rewards of Hybrid #1
    ##@param c: random seed
    ##@param df3a: initialized dataframe
    def hybrid1(self, c, df3a):
        for a in range(self.numberOfTrials):
            cBandit, myAgent, weights, total_episodes, total_reward, init = initializeTensor()
            e = 0.9  # start with highly explorative (90% explore, 10% exploit)
            value = 0.00008  # the value that we will decrement the epsilon
            # Launch the tensorflow graph
            with tf.Session() as sess:
                sess.run(init)
                i = 0
                np.random.seed(c)
                while i < total_episodes:
                    df3a = BanditTensor(cBandit, e, sess, myAgent, weights, total_reward, i, df3a)
                    e -= value  # in the end it would be highly exploitative (10% explore, 90% exploit)
                    i += 1
                c += 1
                print("Trial#", a + 1, ": Hybrid#1 is done!")
        return df3a

    ## Documentation for Hybrid #2 function
    # this would return the rewards of Hybrid #2
    ##@param d: random seed
    ##@param df4a: initialized dataframe
    def hybrid2(self, d, df4a):
        for a in range(self.numberOfTrials):
            cBandit, myAgent, weights, total_episodes, total_reward, init = initializeTensor()
            e = 1  # start with highly explorative (90% explore, 10% exploit)
            value = 0.00018  # the value that we will decrement the epsilon
            # Launch the tensorflow graph
            with tf.Session() as sess:
                sess.run(init)
                i = 0
                np.random.seed(d)
                while i < total_episodes:
                    df4a = BanditTensor(cBandit, e, sess, myAgent, weights, total_reward, i, df4a)
                    e -= value  # in the end it would be highly exploitative (10% explore, 90% exploit)
                    if e < 0.1:  # once it hits at 10% of exploration, it keeps 10% throughout the course.
                        e = 0.1
                    i += 1
                d += 1
                print("Trial#", a + 1, ": Hybrid#2 is done!")
        return df4a

    ## Documentation for Hybrid #3 function
    # this would return the rewards of Hybrid #3
    ##@param e1: random seed
    ##@param df5a: initialized dataframe
    def hybrid3(self, e1, df5a):
        for a in range(self.numberOfTrials):
            cBandit, myAgent, weights, total_episodes, total_reward, init = initializeTensor()
            e = 0.9  # start with highly explorative (90% explore, 10% exploit)
            value = 0.00016  # the value that we will decrement the epsilon
            # Launch the tensorflow graph
            with tf.Session() as sess:
                sess.run(init)
                i = 0
                np.random.seed(e1)
                while i < total_episodes:
                    df5a = BanditTensor(cBandit, e, sess, myAgent, weights, total_reward, i, df5a)
                    e -= value  # in the end it would be highly exploitative (10% explore, 90% exploit)
                    if e < 0.1:  # once it hits at 10% of exploration, it keeps 10% throughout the course.
                        e = 0.1
                    i += 1
                e1 += 1
                print("Trial#", a + 1, ": Hybrid#3 is done!")
        return df5a

    ## Documentation for Hybrid #4 function
    # this would return the rewards of Hybrid #4
    ##@param f: random seed
    ##@param df6a: initialized dataframe
    def hybrid4(self, f, df6a):
        for a in range(self.numberOfTrials):
            cBandit, myAgent, weights, total_episodes, total_reward, init = initializeTensor()
            e = 1  # start with highly explorative (90% explore, 10% exploit)
            value = 0.00036  # the value that we will decrement the epsilon
            # Launch the tensorflow graph
            with tf.Session() as sess:
                sess.run(init)
                i = 0
                np.random.seed(f)
                while i < total_episodes:
                    df6a = BanditTensor(cBandit, e, sess, myAgent, weights, total_reward, i, df6a)
                    e -= value  # in the end it would be highly exploitative (10% explore, 90% exploit)
                    if e < 0.1:  # once it hits at 10% of exploration, it keeps 10% throughout the course.
                        e = 0.1
                    i += 1
                f += 1
                print("Trial#", a + 1, ": Hybrid#4 is done!")
        return df6a

    ## Documentation for Hybrid #5 function
    # this would return the rewards of Hybrid #5
    ##@param g: random seed
    ##@param df7a: initialized dataframe
    def hybrid5(self, g, df7a):
        for a in range(self.numberOfTrials):
            cBandit, myAgent, weights, total_episodes, total_reward, init = initializeTensor()
            e = 0.9  # start with highly explorative (90% explore, 10% exploit)
            value = 0.00032  # the value that we will decrement the epsilon
            # Launch the tensorflow graph
            with tf.Session() as sess:
                sess.run(init)
                i = 0
                np.random.seed(g)
                while i < total_episodes:
                    df7a = BanditTensor(cBandit, e, sess, myAgent, weights, total_reward, i, df7a)
                    e -= value  # in the end it would be highly exploitative (10% explore, 90% exploit)
                    if e < 0.1:  # once it hits at 10% of exploration, it keeps 10% throughout the course.
                        e = 0.1
                    i += 1
                g += 1
                print("Trial#", a + 1, ": Hybrid#5 is done!")
        return df7a
