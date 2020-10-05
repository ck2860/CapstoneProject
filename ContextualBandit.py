import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

import tensorflow.compat.v1 as tf
tf.logging.set_verbosity(tf.logging.ERROR)

adsDF= pd.read_csv('data/Ads_Optimisation.csv')

meansDF = adsDF.mean()
newarr = np.array_split(meansDF, 2)
data = np.array([newarr[0], newarr[1]])
data = np.negative([newarr[0], newarr[1]])

##@file contextualbandit.py
#

## Documentation for Contextual Bandit class
# this class initializes with a state, bandits, number of bandits, and number of actions
class contextual_bandit():
    ## create a new Contextual Bandit
    # set up a contextual bandit
    def __init__(self):
        self.state = 0 # The number of current bandit
        self.bandits = np.array(data)
        #self.bandits = np.random.uniform(low=-10, high=10, size=(4,4))
        self.num_bandits = self.bandits.shape[0] # the bandit number  (3)
        self.num_actions = self.bandits.shape[1] # the action number  (4)

    ## Documentation for getBandit function
    #  Draws the bandit aka state and returns a random state
    def getBandit(self):
        self.state = np.random.randint(0,len(self.bandits))
        return self.state

    ## Documentation for pullArm function
    # Pulls an arm and returns a positive or negative reward
    ##@param action: arm
    def pullArm(self,action):
        #Get a random number.
        bandit = self.bandits[self.state, action]
        result = np.random.randn(1)
        if result > bandit:
            #return a positive reward.
            return 1
        else:
            #return a negative reward.
            return -1