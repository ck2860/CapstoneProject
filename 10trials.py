import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

import tensorflow.compat.v1 as tf
tf.logging.set_verbosity(tf.logging.ERROR)
from ContextualBandit import *
from ContextualBanditAgent import *
from GreedyStrategies import *
from Plot import *

## @file 10trials.py
#
## Documentation for 10trials.py
# 10 trials of evaluation are performed with 10 different random seeds. Epsilon-Decreasing, Epsilon-Greedy, Hybrid#1-#5 are included in the evaluation.
# GreedyStrategies and Plot classes are used here. Please go to see Class tab for more details.

df1a = pd.DataFrame({'x': [], 'y': []})
df2a = pd.DataFrame({'x': [], 'y': []})
df3a = pd.DataFrame({'x': [], 'y': []})
df4a = pd.DataFrame({'x': [], 'y': []})
df5a = pd.DataFrame({'x': [], 'y': []})
df6a = pd.DataFrame({'x': [], 'y': []})
df7a = pd.DataFrame({'x': [], 'y': []})
a1 = 1
b = 1
c = 1
d = 1
f = 1
e1 = 1
g = 1

trial10 = Greedystrategies(10)
df1a = trial10.decreasingEpsilon(a1, df1a)
df2a = trial10.EpsilonGreedy(b, df2a)
df3a = trial10.hybrid1(c, df3a)
df4a = trial10.hybrid2(d, df4a)
df5a = trial10.hybrid3(e1, df5a)
df6a = trial10.hybrid4(f, df6a)
df7a = trial10.hybrid5(g, df7a)
df1aa = pd.DataFrame(df1a['y'].values.reshape(10,20))
df1Ameans = df1aa.mean(0)


df2aa = pd.DataFrame(df2a['y'].values.reshape(10,20))
df2Ameans = df2aa.mean(0)
test2=df2a


df3aa = pd.DataFrame(df3a['y'].values.reshape(10,20))
df3Ameans = df3aa.mean(0)
test3=df3a

df4aa = pd.DataFrame(df4a['y'].values.reshape(10,20))
df4Ameans = df4aa.mean(0)
test4=df4a

df5aa = pd.DataFrame(df5a['y'].values.reshape(10,20))
df5Ameans = df5aa.mean(0)
test5=df5a

df6aa = pd.DataFrame(df6a['y'].values.reshape(10,20))
df6Ameans = df6aa.mean(0)
test6=df6a

df7aa = pd.DataFrame(df7a['y'].values.reshape(10,20))
df7Ameans = df7aa.mean(0)
test7=df7a

eval2 = plot()
eval2.graph(df1a=df1a, df1Ameans=df1Ameans, df2Ameans=df2Ameans, df3Ameans=df3Ameans, df4Ameans=df4Ameans, df5Ameans=df5Ameans, df6Ameans=df6Ameans, df7Ameans=df7Ameans)

print("The Python script is %s" % (sys.argv[0]))
print("10 trials of greedy-based strategies were performed.")
