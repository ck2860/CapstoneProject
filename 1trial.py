import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

import tensorflow.compat.v1 as tf
tf.logging.set_verbosity(tf.logging.ERROR)
from ContextualBandit import *
from ContextualBanditAgent import *
from GreedyStrategies import *
from Plot import *

## @package ONEtrial
## Documentation for 1trial.py
# 1 trial of evaluations is performed with your chosen random seed. Epsilon-Decreasing, Epsilon-Greedy, Hybrid#1-#5 are included in the evaluation.
# GreedyStrategies and Plot classes are used here. Please go to see Class tab for more details.

df1a = pd.DataFrame({'x': [], 'y': []})
df2a = pd.DataFrame({'x': [], 'y': []})
df3a = pd.DataFrame({'x': [], 'y': []})
df4a = pd.DataFrame({'x': [], 'y': []})
df5a = pd.DataFrame({'x': [], 'y': []})
df6a = pd.DataFrame({'x': [], 'y': []})
df7a = pd.DataFrame({'x': [], 'y': []})
number = int(sys.argv[1])
a1 = number
b = number
c = number
d = number
f = number
e1 = number
g = number

trial1 = Greedystrategies(1)
df1a = trial1.decreasingEpsilon(a1, df1a)
df2a = trial1.EpsilonGreedy(b, df2a)
df3a = trial1.hybrid1(c, df3a)
df4a = trial1.hybrid2(d, df4a)
df5a = trial1.hybrid3(e1, df5a)
df6a = trial1.hybrid4(f, df6a)
df7a = trial1.hybrid5(g, df7a)

df1aa = pd.DataFrame(df1a['y'].values.reshape(1, 20))
df1Ameans = df1aa.mean(0)

df2aa = pd.DataFrame(df2a['y'].values.reshape(1, 20))
df2Ameans = df2aa.mean(0)

df3aa = pd.DataFrame(df3a['y'].values.reshape(1, 20))
df3Ameans = df3aa.mean(0)

df4aa = pd.DataFrame(df4a['y'].values.reshape(1, 20))
df4Ameans = df4aa.mean(0)

df5aa = pd.DataFrame(df5a['y'].values.reshape(1, 20))
df5Ameans = df5aa.mean(0)

df6aa = pd.DataFrame(df6a['y'].values.reshape(1, 20))
df6Ameans = df6aa.mean(0)

df7aa = pd.DataFrame(df7a['y'].values.reshape(1, 20))
df7Ameans = df7aa.mean(0)

eval1 = plot()
eval1.graph(df1a=df1a, df1Ameans=df1Ameans, df2Ameans=df2Ameans, df3Ameans=df3Ameans, df4Ameans=df4Ameans,
            df5Ameans=df5Ameans, df6Ameans=df6Ameans, df7Ameans=df7Ameans)


print("The python script is %s" % (sys.argv[0]), "and the random seed is %s" % (sys.argv[1]))