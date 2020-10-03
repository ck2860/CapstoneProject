import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

df= pd.read_csv('StatsTest.csv')
print("T-Test of Epsilon-Greedy and Epsilon-Decreasing:", stats.ttest_ind(df['EGreedy'], df['EDecreasing']))
print("ANOVA of Epsilon-Greedy, Epsilon-Decreasing, and Hybrid#1:", stats.f_oneway(df['EGreedy'], df['EDecreasing'], df['Hybrid1']))
print("ANOVA of Epsilon-Greedy, Epsilon-Decreasing, and Hybrid#2:", stats.f_oneway(df['EGreedy'], df['EDecreasing'], df['Hybrid2']))
print("ANOVA of Epsilon-Greedy, Epsilon-Decreasing, and Hybrid#3:", stats.f_oneway(df['EGreedy'], df['EDecreasing'], df['Hybrid3']))
print("ANOVA of Epsilon-Greedy, Epsilon-Decreasing, and Hybrid#4:", stats.f_oneway(df['EGreedy'], df['EDecreasing'], df['Hybrid4']))
print("ANOVA of Epsilon-Greedy, Epsilon-Decreasing, and Hybrid#5:", stats.f_oneway(df['EGreedy'], df['EDecreasing'], df['Hybrid5']))

df1= pd.read_csv('TukeyAnalysis.csv')
tukey = pairwise_tukeyhsd(endog=df1['MeanRewards'], groups=df1['Strategy'], alpha=0.05)
tukey.plot_simultaneous()
plt.vlines(x=49.57,ymin=-0.5,ymax=4.5)
plt.show()