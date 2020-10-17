import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
from addData import *
import tensorflow.compat.v1 as tf
tf.logging.set_verbosity(tf.logging.ERROR)

## Documentation for Contextual Bandit class
# this class initializes with a state, bandits, number of bandits, and number of actions.
# It has two functions: getBandit(): and pullArm(action):

class contextual_bandit():
    ## Create a new Contextual Bandit enviroment that our agent will interact with
    # setting up a contextual bandit
    def __init__(self):
        self.state = 0  # The number of current bandit
        data = importingData() # importing data (ad selections)
        self.bandits = np.array(data)  # the data is being used as bandits/arms.
        self.num_bandits = self.bandits.shape[0]  # the bandit number
        self.num_actions = self.bandits.shape[1]  # the action number

    ## Documentation for getBandit function
    #  Draws the bandit aka state and returns a random state
    def getBandit(self):
        self.state = np.random.randint(0, len(self.bandits))  # selecting a random state.
        return self.state  # return a state

    ## Documentation for pullArm function
    # Pulls an arm and returns a positive or negative reward
    ##@param action: arm
    def pullArm(self, action):
        bandit = self.bandits[self.state, action]  # Get a random number.
        result = np.random.randn(1)  # a sample from standard normal distribution
        if result > bandit:
            # return a positive reward.
            return 1
        else:
            # return a negative reward.
            return -1
