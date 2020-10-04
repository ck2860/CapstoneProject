Main Page {#mainpage}
=========

# Contextual Bandit Methods

We would solve contextual bandit problems, using a policy-gradient based reinforcement learning. We would evaluate seven different epsilon-based strategies. We have Epsilon-Greedy, Epsilon-Decreasing, and five different combinations of Epsilon-Greedy and Epsilon-Decreasing. We tweaked with the epsilon probability and episodes.  Five combinations were created! 

In the experiment, we have 10,000 episodes and use the learning rate of 0.05. The epsilon is the probability of exploration.

**Epsilon-Decreasing:** The experiment starts with pure exploration and epsilon is decreased to 0% in the end.

**Epsilon-Greedy:** Epsilon is 10%. The probability is fixed.

**Hybrid#1:** Epsilon is decreased from 90% to 10% throughout the experiment. 

**Hybrid#2:** Epsilon is decreased from 100% to 10% in the first 5,000 episodes and keeps as 10% for the rest of the experiment.

**Hybrid#3:** Epsilon is decreased from 90% to 10% in the first 5,000 episodes and keeps as 10% for the rest of the experiment.

**Hybrid#4:** Epsilon is decreased from 100% to 10% in the first 2,500 episodes and keeps as 10% for the rest of the experiment.

**Hybrid#5:** Epsilon is decreased from 90% to 10% in the first 2,500 episodes and keeps as 10% for the rest of the experiment.

All of the strategies are in the evaluation code. The reinforcement learning code is derived from [Md. Rezaul Karim](https://www.oreilly.com/library/view/tensorflow-powerful-predictive/9781789136913/) and [Arthur Juliani](https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-1-5-contextual-bandits-bff01d1aad9c). 

## Table of contents
* Requirements
* Setup
* Instructions

## Requirements
You will need to have a couple of Python's libraries/packages: numPy, pandas, Tensorflow, matplotlibpyplot, scipy.stats, and atsmodels.stats.multicomp. Most of the packages are pre-installed in Anaconda.

You will be able to run the programs in Anaconda prompt. 

*Note: if you are able to run the programs or have those packages installed in a different software or environment. You can skip the Setup section or  (You could install the packages with Pip.)*
                                            
## Setup
You will be installing [Anaconda](http://anaconda.com/downloads) and required packages for this project. 
Please download the right version for your system/computer.
 
Recall that numPy, pandas,  matplotlibpyplot, and stats packages are already installed in Jupyter notebook, you only need to install TensorFlow. Using the Anaconda Prompt, it can be done by entering:

```
$ conda install -c conda-forge tensorflow
```
Note: It depends on your system. There are other commands that you can run with conda. Please read more about [Tensorflow Installment](https://anaconda.org/conda-forge/tensorflow). 


## Instructions

There are two options that you could download the whole code. 
1. Click "Code" green button and "download ZIP" on the top right hand on the repository page. 
2. Clone a repository by performing git clone. 

Once you have the files in your system/computer, you should see there are three different datasets: [Ads Optimisation](https://github.com/ck2860/MidtermCode-CondyKan/blob/master/data/Ads_Optimisation.csv), [Mean Rewards Results](https://github.com/ck2860/MidtermCode-CondyKan/blob/master/data/MeanRewardsResult.csv), and [Tukey Data](https://github.com/ck2860/MidtermCode-CondyKan/blob/master/data/TukeyData.csv), which are already provided in the data folder. You will be using them for this code project so make sure they are in the right folder -- data. When you run the python scripts, they should be able to find the datasets in the data folder. 
 
 The Ads Optimisation data set is obtained from [Kaggle](https://www.kaggle.com/akram24/ads-ctr-optimisation).  You will be using this for reinforcement learning and evaluations. 
 
Both Mean Rewards Result and Tukey datasets are the results from the evaluations with 20 random seeds. The Mean Rewards Result data is used for T-tests and ANOVA. Lastly, the Tukey Data is used for Tukey Test. 

*Please make sure they all are in the same directory so the scripts
would be able to recognize the data sets from the data folder. For this reinforcement learning, the data set can be used in online advertising to determine which is the best ad to show the user. It does not mean they would work with other bandit problems.*


You will have five Pythons scripts to run if you like! The figures for evaluations and Tukey should be in pop-up windows. For evaluations, there are three Python program that you can run: [1trial.py](https://github.com/ck2860/MidtermCode-CondyKan/blob/master/1trial.py), [10trials.py](https://github.com/ck2860/MidtermCode-CondyKan/blob/master/10trials.py), and [20trials.py](https://github.com/ck2860/MidtermCode-CondyKan/blob/master/20trials.py). 1trial.py runs one trial of evaluation. 10trials.py runs 10 trials of evaluation. Then 20trials.py runs 20 trials of evaluation. Last two Python programs: [StatsTests.py](https://github.com/ck2860/MidtermCode-CondyKan/blob/master/StatsTests.py) and [Tukey.py](https://github.com/ck2860/MidtermCode-CondyKan/blob/master/Tukey.py) are used for statistical analysis. You run T-tests and ANOVA by running StatsTest.py. You also can do a Tukey test by compiling Tukey.py. Please read [documentation](https://ck2860.github.io/MidtermCode-CondyKan/) for more details. 

Please open your Anaconda Prompt and go to the directory where you downloaded the project code.


If you want to run one trial of greedy-based strategies of your chosen random seed. For example, if you want to run it with a random seed of 5. You can compile it by:
```
python 1trial.py 5
```
*Note that if you use a different random seed for 1trial.py, the evaluation results may be different due to reinforcement learning and strategies.*

If you want to run 10 trials of greedy-based strategies of your chosen random seed. You should compile it by:
```
python 10trials.py
```

If you want to run 20 trials of greedy-based strategies of your chosen random seed. Your command line should be:
```
python 20trials.py
```
For t-tests and ANOVA, you would want to run by:

```
python StatsTests.py
```

Lastly, you can run a Tukey Test by:
```
python Tukey.py
```


If you have any questions, please feel free to email me at ck2860@rit.edu. 

Thanks for reading!
