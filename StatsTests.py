import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

##@package StatsTest
#Documentation for StatsTest.py
#
# we would be using t-tests and ANOVA for analysis.
df= pd.read_csv('data/StatsTest.csv')
print("T-Test of Epsilon-Greedy and Epsilon-Decreasing:", stats.ttest_ind(df['EGreedy'], df['EDecreasing']))
print("ANOVA of Epsilon-Greedy, Epsilon-Decreasing, and Hybrid#1:", stats.f_oneway(df['EGreedy'], df['EDecreasing'], df['Hybrid1']))
print("ANOVA of Epsilon-Greedy, Epsilon-Decreasing, and Hybrid#2:", stats.f_oneway(df['EGreedy'], df['EDecreasing'], df['Hybrid2']))
print("ANOVA of Epsilon-Greedy, Epsilon-Decreasing, and Hybrid#3:", stats.f_oneway(df['EGreedy'], df['EDecreasing'], df['Hybrid3']))
print("ANOVA of Epsilon-Greedy, Epsilon-Decreasing, and Hybrid#4:", stats.f_oneway(df['EGreedy'], df['EDecreasing'], df['Hybrid4']))
print("ANOVA of Epsilon-Greedy, Epsilon-Decreasing, and Hybrid#5:", stats.f_oneway(df['EGreedy'], df['EDecreasing'], df['Hybrid5']))
