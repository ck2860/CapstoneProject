import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

import tensorflow.compat.v1 as tf
tf.logging.set_verbosity(tf.logging.ERROR)

##@file Plot.py
#

## Documentation for plot class
# this class has a graph function that you will be using for evaluations.
class plot():
    ## Documentation for graph function
    # Plot the x and y values for graphing and returns the graph of evaluation
    ##@param a1: random seed
    ##@parm df1a: initialized dataframe
    def graph(self, df1a, df1Ameans, df2Ameans, df3Ameans, df4Ameans, df5Ameans, df6Ameans, df7Ameans):
        df=pd.DataFrame({'x': df1a['x'], 'Epsilon-Decreasing': df1Ameans, 'Epsilon-Greedy': df2Ameans, 'Hybrid #1': df3Ameans, 'Hybrid #2': df4Ameans , 'Hybrid#3': df5Ameans, 'Hybrid#4': df6Ameans, 'Hybrid#5':df7Ameans})

        # Initialize the figure
        plt.style.use('seaborn-darkgrid')

        # my_dpi=96
        plt.figure(figsize=(900/96, 900/96), dpi=96)
        # create a color palet6e
        palette = plt.get_cmap('tab10')

        # multiple line plot
        num=0
        for column in df.drop('x', axis=1):
            num+=1

            # Find the right spot on the plot
            plt.subplot(3,3, num)

            # plot every groups, but discreet
            for v in df.drop('x', axis=1):
                plt.plot(df['x'], df[v], marker='', color='black', linewidth=0.5, alpha=0.5)

            # Plot the lineplot
            plt.plot(df['x'], df[column], marker='', color=palette(num), linewidth=2.4, alpha=0.9, label=column)

            # Same limits for everybody!
            plt.xlim(0,10000)
            plt.ylim(-8,200)

            # Not ticks everywhere
            if num in range(5) :
                plt.tick_params(labelbottom=False)
            if num in [2,3,5,6] :
                plt.tick_params(labelleft=False)

                #Adding labels
            if num in range(1,8,3):
                plt.ylabel("Mean Reward",fontsize=10)
            if num in [5,6,7] :
                plt.xlabel("Episode",fontsize=10)

            # Add title
            plt.title(column, loc='left', fontsize=12, fontweight=0, color=palette(num) )
            # general title
            plt.suptitle("In comparison of mean rewards", fontsize=13, fontweight=0, color='black', style='italic', y=1.02)
        return plt.show()
