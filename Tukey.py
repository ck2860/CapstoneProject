import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

df1= pd.read_csv('data/TukeyData.csv')
tukey = pairwise_tukeyhsd(endog=df1['MeanRewards'], groups=df1['Strategy'], alpha=0.05)
tukey.plot_simultaneous()
plt.vlines(x=49.57,ymin=-0.5,ymax=4.5)
plt.show()