from unittest import TestCase
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
import tensorflow.compat.v1 as tf
from ContextualBandit import *
from ContextualBanditAgent import *
# tf.logging.set_verbosity(tf.logging.ERROR)
from GreedyStrategies import *


## @package test_Greedystrategies
## Documentation for test_GreedyStrategies.py
# We test decreasing-epsilon, greedy-epsilon, hybrid#1, hybrid#2, hybrid#3, hybrid#4, and hybrid#5 functions from GreedyStrategies.py with expected results.
# These tests help us to ensure that our network is working and consistent. With these tests, we can know our single-layer perceptron works.
# Since all functions from addData.py, ContextualBandit.py, ContexutualBanditAgent.py, and InitializeTensor.py are utilized in the Greedy-based strategies, we test all functions at once!
# We will have seven tests with different random seed each and match with expected results in one trial.

class TestGreedystrategies(TestCase):
    ## Documentation for test_decreasing_epsilon function
    #  the random seed is 8 and the 8th value of mean rewards result is expectedlly 60.8
    def test_decreasing_epsilon(self):
        trial1 = greedyStrategies(1)
        df1a = pd.DataFrame({'x': [], 'y': []})
        a1 = 8
        df1a = trial1.decreasingEpsilon(a1, df1a)
        df1aa = pd.DataFrame(df1a['y'].values.reshape(1, 20))
        df1Ameans = df1aa.mean(0)
        self.assertEqual(df1Ameans[8], 60.8)

    ## Documentation for test_epsilon_greedy function
    #  the random seed is 2 and the 8th value of mean rewards result is expectedlly 40.4
    def test_epsilon_greedy(self):
        trial1 = greedyStrategies(1)
        df1a = pd.DataFrame({'x': [], 'y': []})
        a1 = 2
        df1a = trial1.epsilonGreedy(a1, df1a)
        df1aa = pd.DataFrame(df1a['y'].values.reshape(1, 20))
        df1Ameans = df1aa.mean(0)
        self.assertEqual(df1Ameans[8], 40.4)

    ## Documentation for test_hybrid1 function
    #  the random seed is 3 and the 8th value of mean rewards result is expectedlly 68.4
    def test_hybrid1(self):
        trial1 = greedyStrategies(1)
        df1a = pd.DataFrame({'x': [], 'y': []})
        a1 = 3
        df1a = trial1.hybrid1(a1, df1a)
        df1aa = pd.DataFrame(df1a['y'].values.reshape(1, 20))
        df1Ameans = df1aa.mean(0)
        self.assertEqual(df1Ameans[8], 68.4)

    ## Documentation for test_hybrid2 function
    #  the random seed is 5 and the 8th value of mean rewards result is expectedlly 68.2
    def test_hybrid2(self):
        trial1 = greedyStrategies(1)
        df1a = pd.DataFrame({'x': [], 'y': []})
        a1 = 5
        df1a = trial1.hybrid2(a1, df1a)
        df1aa = pd.DataFrame(df1a['y'].values.reshape(1, 20))
        df1Ameans = df1aa.mean(0)
        self.assertEqual(df1Ameans[8], 68.2)

    ## Documentation for test_hybrid3 function
    #  the random seed is 8 and the 8th value of mean rewards result is expectedlly 49.0
    def test_hybrid3(self):
        trial1 = greedyStrategies(1)
        df1a = pd.DataFrame({'x': [], 'y': []})
        a1 = 8
        df1a = trial1.hybrid3(a1, df1a)
        df1aa = pd.DataFrame(df1a['y'].values.reshape(1, 20))
        df1Ameans = df1aa.mean(0)
        self.assertEqual(df1Ameans[8], 49.0)

    ## Documentation for test_hybrid4 function
    #  the random seed is 6 and the 8th value of mean rewards result is expectedlly 65.2
    def test_hybrid4(self):
        trial1 = greedyStrategies(1)
        df1a = pd.DataFrame({'x': [], 'y': []})
        a1 = 6
        df1a = trial1.hybrid4(a1, df1a)
        df1aa = pd.DataFrame(df1a['y'].values.reshape(1, 20))
        df1Ameans = df1aa.mean(0)
        self.assertEqual(df1Ameans[8], 65.2)

    ## Documentation for test_hybrid5 function
    #  the random seed is 7 and the 8th value of mean rewards result is expectedlly 55.6
    def test_hybrid5(self):
        trial1 = greedyStrategies(1)
        df1a = pd.DataFrame({'x': [], 'y': []})
        a1 = 7
        df1a = trial1.hybrid5(a1, df1a)
        df1aa = pd.DataFrame(df1a['y'].values.reshape(1, 20))
        df1Ameans = df1aa.mean(0)
        self.assertEqual(df1Ameans[8], 55.6)
