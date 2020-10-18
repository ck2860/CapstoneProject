import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import tensorflow.compat.v1 as tf
tf.logging.set_verbosity(tf.logging.ERROR)
from ContextualBandit import *
from ContextualBanditAgent import *
from GreedyStrategies import *
from Plot import *
from addData import *

## Documentation for Evaluation.py
# n trial(s) of evaluations is/are performed with your chosen random seed. Epsilon-Decreasing, Epsilon-Greedy, Hybrid#1-#5 are included in the evaluation.
# n is determined by the person's input
# GreedyStrategies and Plot classes are used here. Please go to see Class tab for more details.

# This is when if the person fails to provide a number of trials on the command line.
if len(sys.argv) < 2:
    print("You failed to provide a number of trials as input on the command line.")
    sys.exit(1)

df = pd.DataFrame({'x': [], 'y': []})  # initialize a dataframe for the graph
number1 = sys.argv[1]  # the number from the command line

seed = 1  # the random seed number
numberOfTrials = get_integer(number1)  # We call the get_integer function to ensure that the number from the command line is an integer.
intNum = int(numberOfTrials)  # then cast it to integer after it returns from get_integer function

trial = greedyStrategies(intNum)  # instantiating the greedyStrategies class
df1a = trial.decreasingEpsilon(seed, df)  # performing epsilon decreasing function
df2a = trial.epsilonGreedy(seed, df)  # performing epsilon greedy function
df3a = trial.hybrid1(seed, df)  # performing hybrid#1 function
df4a = trial.hybrid2(seed, df)  # performing hybrid#2 function
df5a = trial.hybrid3(seed, df)  # performing hybrid#3 function
df6a = trial.hybrid4(seed, df)  # performing hybrid#4 function
df7a = trial.hybrid5(seed, df)  # performing hybrid#5 function

df1aa = pd.DataFrame(df1a['y'].values.reshape(intNum, 20))
df1Ameans = df1aa.mean(0)  # averaging the rewards from epsilon decreasing strategy

df2aa = pd.DataFrame(df2a['y'].values.reshape(intNum, 20))
df2Ameans = df2aa.mean(0)  # averaging the rewards from epsilon greedy strategy

df3aa = pd.DataFrame(df3a['y'].values.reshape(intNum, 20))
df3Ameans = df3aa.mean(0)  # averaging the rewards from hybrid#1 strategy

df4aa = pd.DataFrame(df4a['y'].values.reshape(intNum, 20))
df4Ameans = df4aa.mean(0)  # averaging the rewards from hybrid#2 strategy

df5aa = pd.DataFrame(df5a['y'].values.reshape(intNum, 20))
df5Ameans = df5aa.mean(0)  # averaging the rewards from hybrid#3 strategy

df6aa = pd.DataFrame(df6a['y'].values.reshape(intNum, 20))
df6Ameans = df6aa.mean(0)  # averaging the rewards from hybrid#4 strategy

df7aa = pd.DataFrame(df7a['y'].values.reshape(intNum, 20))
df7Ameans = df7aa.mean(0)  # averaging the rewards from hybrid#5 strategy

# Calling Eval_graph function to make the line graphs. All of the rewards from the epsilon-based strategies are plotted.
Eval_graph(df1a=df1a, df1Ameans=df1Ameans, df2Ameans=df2Ameans, df3Ameans=df3Ameans, df4Ameans=df4Ameans, df5Ameans=df5Ameans, df6Ameans=df6Ameans, df7Ameans=df7Ameans)

print("The python script is %s" % (sys.argv[0]), "and  %s trial(s) of epsilon-based strategies is/are done." % (sys.argv[1]))


