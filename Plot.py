import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import tensorflow.compat.v1 as tf
tf.logging.set_verbosity(tf.logging.ERROR)


## Documentation for graph function
# Plot the x and y values for graphing and returns the graph of evaluation
##@param df1a: any dataframe
##@param df1Ameans: Decreasing-Epsilon means rewards
##@param df2Ameans: Greedy-Epsilon means rewards
##@param df3Ameans: Hybrid#1 means rewards
##@param df4Ameans: Hybrid#2 means rewards
##@param df5Ameans: Hybrid#3 means rewards
##@param df6Ameans: Hybrid#4 means rewards
##@param df7Ameans: Hybrid#5 means rewards
def Eval_graph(df1a, df1Ameans, df2Ameans, df3Ameans, df4Ameans, df5Ameans, df6Ameans, df7Ameans):
    df = pd.DataFrame({'x': df1a['x'], 'Epsilon-Decreasing': df1Ameans, 'Epsilon-Greedy': df2Ameans,
                           'Hybrid #1': df3Ameans, 'Hybrid #2': df4Ameans, 'Hybrid#3': df5Ameans,
                           'Hybrid#4': df6Ameans, 'Hybrid#5': df7Ameans})

    plt.style.use('seaborn-darkgrid')

    plt.figure(figsize=(900 / 96, 900 / 96), dpi=96)
    palette = plt.get_cmap('tab10')
    num = 0
    for column in df.drop('x', axis=1):
        num += 1
        plt.subplot(3, 3, num)
        for v in df.drop('x', axis=1):
            plt.plot(df['x'], df[v], marker='', color='black', linewidth=0.5, alpha=0.5)

        # Plotting
        plt.plot(df['x'], df[column], marker='', color=palette(num), linewidth=2.4, alpha=0.9, label=column)
        plt.xlim(0, 10000)
        plt.ylim(-8, 200)

        # Removing some ticks
        if num in range(5):
            plt.tick_params(labelbottom=False)
        if num in [2, 3, 5, 6]:
            plt.tick_params(labelleft=False)

        # Adding labels
        if num in range(1, 8, 3):
            plt.ylabel("Mean Reward", fontsize=10)
        if num in [5, 6, 7]:
            plt.xlabel("Episode", fontsize=10)

        # Adding title
        plt.title(column, loc='left', fontsize=12, fontweight=0, color=palette(num))
        plt.suptitle("In comparison of mean rewards", fontsize=13, fontweight=0, color='black', style='italic', y=1.02)
    return plt.show()


## Documentation for tukey function
# Plot the Tukey post hoc test
def tukey_plot(result):
    # perform plot_simultaneous function from the statsmodel package
    result.plot_simultaneous()
    return plt.show()
