import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import warnings
warnings.filterwarnings("ignore")

##@file Tukey.py
#Documentation for Tukey.py
# we use a Tukey post hoc analysis to confirm where the differences occurred between greedy-based strategies: Epsilon-Greedy, Epsilon-Decreasing, Hybrid#2, and Hybrid#4.
# Statsmodels.stats.multicomp package is used; only pairwise_tukeyhsd function is performed.
df1= pd.read_csv('data/TukeyData.csv')
# importing a dataset
tukey = pairwise_tukeyhsd(endog=df1['MeanRewards'], groups=df1['Strategy'], alpha=0.05)
tukey.plot_simultaneous()
plt.show()