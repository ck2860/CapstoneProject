from unittest import TestCase
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
import tensorflow.compat.v1 as tf
from ContextualBandit import *
from ContextualBanditAgent import *
# tf.logging.set_verbosity(tf.logging.ERROR)
from GreedyStrategies import *


class TestGreedystrategies(TestCase):
    def test_decreasing_epsilon(self):
        trial1 = greedyStrategies(1)
        df1a = pd.DataFrame({'x': [], 'y': []})
        a1 = 8
        df1a = trial1.decreasingEpsilon(a1, df1a)
        df1aa = pd.DataFrame(df1a['y'].values.reshape(1, 20))
        df1Ameans = df1aa.mean(0)
        self.assertEqual(df1Ameans[8], 60.8)

    def test_epsilon_greedy(self):
        trial1 = greedyStrategies(1)
        df1a = pd.DataFrame({'x': [], 'y': []})
        a1 = 8
        df1a = trial1.epsilonGreedy(a1, df1a)
        df1aa = pd.DataFrame(df1a['y'].values.reshape(1, 20))
        df1Ameans = df1aa.mean(0)
        self.assertEqual(df1Ameans[8], 93.2)

    def test_hybrid1(self):
        trial1 = greedyStrategies(1)
        df1a = pd.DataFrame({'x': [], 'y': []})
        a1 = 8
        df1a = trial1.hybrid1(a1, df1a)
        df1aa = pd.DataFrame(df1a['y'].values.reshape(1, 20))
        df1Ameans = df1aa.mean(0)
        self.assertEqual(df1Ameans[8], 59.4)

    def test_hybrid2(self):
        trial1 = greedyStrategies(1)
        df1a = pd.DataFrame({'x': [], 'y': []})
        a1 = 8
        df1a = trial1.hybrid2(a1, df1a)
        df1aa = pd.DataFrame(df1a['y'].values.reshape(1, 20))
        df1Ameans = df1aa.mean(0)
        self.assertEqual(df1Ameans[8], 56.6)

    def test_hybrid3(self):
        trial1 = greedyStrategies(1)
        df1a = pd.DataFrame({'x': [], 'y': []})
        a1 = 8
        df1a = trial1.hybrid3(a1, df1a)
        df1aa = pd.DataFrame(df1a['y'].values.reshape(1, 20))
        df1Ameans = df1aa.mean(0)
        self.assertEqual(df1Ameans[8], 49.0)

    def test_hybrid4(self):
        trial1 = greedyStrategies(1)
        df1a = pd.DataFrame({'x': [], 'y': []})
        a1 = 8
        df1a = trial1.hybrid4(a1, df1a)
        df1aa = pd.DataFrame(df1a['y'].values.reshape(1, 20))
        df1Ameans = df1aa.mean(0)
        self.assertEqual(df1Ameans[8], 59.2)

    def test_hybrid5(self):
        trial1 = greedyStrategies(1)
        df1a = pd.DataFrame({'x': [], 'y': []})
        a1 = 8
        df1a = trial1.hybrid5(a1, df1a)
        df1aa = pd.DataFrame(df1a['y'].values.reshape(1, 20))
        df1Ameans = df1aa.mean(0)
        self.assertEqual(df1Ameans[8], 80.0)
