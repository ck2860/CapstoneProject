Main Page {#mainpage}
=========

# Contextual Bandit Methods
We would solve contextual bandit problems, using a policy-gradient based reinforcement learning. We would evaluate seven different epsilon-based strategies. We have Epsilon-Greedy, Epsilon-Decreasing, and five different combinations of Epsilon-Greedy and Epsilon-Decreasing. We tweaked with the epsilon probability and episodes.  Five combinations were created! 

In the experiment, we have 10,000 episodes and use the learning rate of 0.05. The epsilon is the probability of exploration.

**Epsilon-Decreasing:** The experiment starts with pure exploration and epsilon is decreased then becomes 0 at the end.

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
* Diagrams
* Instructions
* Unit Tests

## Requirements
Since all the code are written in Python, you should have or download [Python](https://www.python.org/downloads/).  Your Python version should be 3.5-3.8.

You will need to have a couple of Python's libraries/packages: numPy, pandas, Tensorflow, matplotlibpyplot, scipy.stats, and atsmodels.stats.multicomp. Most of the packages are pre-installed in Anaconda.

You will be able to run the programs in Anaconda prompt. 

*Note: if you are able to run the programs or have those packages installed in a different software or environment. Make sure your version of tensorflow is 1.14. You can skip the Setup section or  (You could install the packages with Pip.)*
                                            
## Setup
You will be installing [Anaconda](http://anaconda.com/downloads) and required packages for this project. 
Please download the right version for your system/computer.
 
Recall that numPy, pandas,  matplotlibpyplot, and stats packages are already installed in Anaconda, you only need to install TensorFlow 1.14. Using the Anaconda Prompt, it can be done by entering:

```
conda install -c conda-forge tensorflow=1.14
```
Note: It depends on your system. There are other commands that you can run with conda. Please read more about [Tensorflow Installment](https://anaconda.org/conda-forge/tensorflow). 

## Diagrams
The two diagrams are in [Models](https://github.com/ck2860/MidtermCode-CondyKan/tree/master/models).

**1.)** [Domain Driven Diagram](https://github.com/ck2860/MidtermCode-CondyKan/blob/master/models/DomainModel-Midterm.png) describes the development of reinforcement learning and greedy-based strategies.  

**2.)** [Sequence Diagram](https://github.com/ck2860/MidtermCode-CondyKan/blob/master/models/SequenceDiagram-Midterm.png) explains how the program performs in the sequential order. 

## Instructions

There are two options that you could download the whole code. 

1.) Clicking "Code" green button and "download ZIP" on the top right hand on the Git repository page. 

2.) Cloning the Git repository by performing git clone. 

Once you have the files in your system/computer, you should see there are three different datasets: [Ads Optimisation](https://github.com/ck2860/MidtermCode-CondyKan/blob/master/data/Ads_Optimisation.csv), [Mean Rewards Results](https://github.com/ck2860/MidtermCode-CondyKan/blob/master/data/MeanRewardsResult.csv), and [Tukey Data](https://github.com/ck2860/MidtermCode-CondyKan/blob/master/data/TukeyData.csv), which are already provided in the data folder. You will be using them for this code project so make sure they are in the right folder -- data. When you run the python codes, they should be able to find the datasets in the data folder. 
 
The Ads Optimisation data set is obtained from [Kaggle](https://www.kaggle.com/akram24/ads-ctr-optimisation).  You
will be using this for reinforcement learning and evaluations. For the evaluation, the ads are narrowed down to 5
ads so our reinforcement agent has five ads (Ad#0 - Ad#4) to select.
 
Both MeanRewardsResult and TukeyData datasets are the results from the evaluations with 20 random seeds (#1-#20). The Mean Rewards Result data is used for T-tests and ANOVA. Lastly, the TukeyData is used for the Tukey's Post Hoc Test. 

*Please make sure they all are in the same directory so the codes would be able to recognize the data sets from the data folder. For this reinforcement learning, the data set can be used in online advertising to determine which is the best ad to show the user. It does not mean they would work with other bandit problems.*

We have classes and functions in the folder that would be utilized in the running codes. [addData](https://github.com/ck2860/MidtermCode-CondyKan/blob/master/addData.py), [ContextualBandit](https://github.com/ck2860/MidtermCode-CondyKan/blob/master/ContextualBandit.py), [ContextualBanditAgent](https://github.com/ck2860/MidtermCode-CondyKan/blob/master/ContextualBanditAgent.py), [InitializeTensor](https://github.com/ck2860/MidtermCode-CondyKan/blob/master/InitializeTensor.py) and [GreedyStrategies](https://github.com/ck2860/MidtermCode-CondyKan/blob/master/GreedyStrategies.py) are for the evaluations. It would import data, set contextual bandits up, initialize the network, run reinforcement learning, and test greedy-based strategies. Finally, [Plot](https://github.com/ck2860/MidtermCode-CondyKan/blob/master/Plot.py) is programmed to plot evaluation results. You have two compilable Pythons codes in the folder, you could run them if you like! 


Please open your Anaconda Prompt and go to the directory where you downloaded the project code. Please read the directions below before you run the code. Note that the ```$``` is not part of the command. You may get warning messages due to the packages that are used in the code. 

**1.)**  For evaluations, you may compile [Evaluation](https://github.com/ck2860/MidtermCode-CondyKan/blob/master/Evaluation.py) with your chosen number of trials. The figure for evaluations should be in a pop-up window.

There will be two arguments in your command line: the first one is the name of python file (Evaluation.py) and second one is the number of trials. For example, if you want to do 5 trials (it may take up to 5 minutes), you can run the line below: 
```
$ python Evaluation.py 5
```
*Note that if you decide to run more than one trial, it may take longer. The results may be different due to number of trials. *

**2.)**  For statistical analysis, we analyze the results from the 20 trials evaluations. You can perform T-tests, ANOVA, and Tukey Post Hoc tests by compiling [StatsTests](https://github.com/ck2860/MidtermCode-CondyKan/blob/master/StatsTests.py). After narrowing them down and performing the Post Hoc test, you may see the differences occurred between greedy-based strategies. The Tukey post hoc plot should be in a pop-up window. 

You will only have one command-line argument: StatTests.py. You would want to run by:
```
$ python StatsTests.py
```

## Unit Tests
For unit testings, we have three files to run. Note that you may get warning messages due to the packages that are used in the code but you can ignore them. 

**1.)**  [test_addData](https://github.com/ck2860/MidtermCode-CondyKan/blob/master/test_addData.py) is used to ensure that the function handle exceptions properly. Recall that we provide a number of trials when we compile Evaluation program code, we could not have a non-integer value in the command line.  In this file, we have five tests that are expected to pass. You can compile it by:

```
$ python -m unittest test_addData.py
```

**2.)**  [test_Greedystrategies](https://github.com/ck2860/MidtermCode-CondyKan/blob/master/test_GreedyStrategies.py) is for testing decreasing-epsilon, greedy-epsilon, hybrid#1, hybrid#2, hybrid#3, hybrid#4, and hybrid#5 functions.One trial of each function would be performed. Since all the functions from addData.py, ContextualBandit.py, ContexutualBanditAgent.py, and InitializeTensor.py are used in the epsilon-based strategy functions, we test all the functions at once. Here, we have seven tests with varied random seed and match them with expected results. With the tests, we can confirm that the network is consistent. You would want to run by:

```
$ python -m unittest test_GreedyStrategies.py
```

**3.)**  [test_InitalizeTensor](https://github.com/ck2860/MidtermCode-CondyKan/blob/master/test_InitializeTensor.py) is used to check and ensure that the types of parameters are correct. We have three tests to perform. You can compile it by: 

```
$ python -m unittest test_InitializeTensor.py
```

*You could read [documentation](https://ck2860.github.io/MidtermCode-CondyKan/) for more details.* If you have any questions, please feel free to email me at ck2860@rit.edu. 

Thanks for reading!