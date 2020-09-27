# Contextual Bandits (HII)
We would solve contextual bandit problems, using a policy-gradient based reinforcement learning. We would evaluate three different greedy strategies: Explore-First, Epsilon-Decreasing, and Epsilon-Greedy. 

The reinforcement learning code is derived from [Md. Rezaul Karim](https://www.oreilly.com/library/view/tensorflow-powerful-predictive/9781789136913/) and [Arthur Juliani](https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-1-5-contextual-bandits-bff01d1aad9c). 

## Table of contents
* [Requirements](#Requirements)
* [Setup](#Setup)
* [Instructions](#Instructions)

## Requirements
This code submission run in the Jupyter Notebook, so you may need to install Jupyter Notebook. Also, the code is in Python so you will need to use few Python's libraries/packages: numPy, pandas, TensorFlow, and Matplotib. 
 
Note: if you are able to run the code in a different software/environment, you can skip the Setup section. You could install numPy, pandas, TensorFlow, and Matplotlib with with Pip.
                                                                               

## Setup
To run this project, install [Jupyter Notebook](https://jupyter.org/). If you have Pip installed in your environment, you could run and install it by entering: 

```
$ pip install jupyter
```

Once you get your Jupyter Notebook installed in your operating system, you can launch it by entering:

```
$ jupyter notebook
```

![Jupyter Notebook Dashboard](https://github.com/ck2860/MidtermCode-CondyKan/blob/master/screenshot/jupyternotebook.png?raw=true)

It will take you to the Notebook Dashboard. To create a notebook (the notebook is where you run code), you can click the "New" drop-down button on top-right then select "Python 3". Finally, you are in your first notebook then may run code. 

numPy and Matplotlib are already installed in Jupyter notebook, you only need to install TensorFlow. Using the Anaconda Prompt, it can be done by entering:

```
$ conda install -c conda-forge tensorflow
```
Note: It depends on your system. There are other commands that you can run with conda. Please read more about the [Tensorflow Installment](https://anaconda.org/conda-forge/tensorflow). 

## Instructions

At first, you will need to download [Ads Optimisation data](https://github.com/ck2860/CodeSubmissionS2/blob/master/data/Ads_Optimisation.csv), which is already provided in the data folder. You will import the data set in the Notebook. The data set is obtained from [Kaggle](https://www.kaggle.com/akram24/ads-ctr-optimisation). You will use pandas and numpy as well. 

Note: For this reinforcement learning, the data set can be used in online advertising to determine which is the best ad to show the user. It does not mean they would work with other bandit problems. 

```
import pandas as pd
import numpy as np 
dir ='your directory/'
file = 'Ads_Optimisation.csv'
adsDF = pd.read_csv(dir+file)
adsDF.head(2)
```
Then you will need to find the probability of each ad. 10 ads will be split into 2 arrays. You will have two bandits. 

```
meansDF = adsDF.mean()
newarr = np.array_split(meansDF, 2)
data = np.array([newarr[0], newarr[1]])
data = np.negative([newarr[0], newarr[1]])
data
```
Your output data should look like this: 

![Data Array](https://github.com/ck2860/MidtermCode-CondyKan/blob/master/screenshot/meansData.png?raw=true)

Now, you can compile exploreFirst, epsilonDecreasing, and eGreedy strategies in the repo. Additionally, there are three different learning rates (0.001, 0.005, and 0.005) that you could test and compare the results. Because of the reinforcement learning and strategies, the evaluation results may be different. 



Feel free to email me at ck2860@rit.edu. 

Thanks for reading!



