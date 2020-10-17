from unittest import TestCase
from InitializeTensor import *
import ContextualBanditAgent
import ContextualBandit


class Test(TestCase):
    def test_initialize_tensor1(self):
        cBandit, myAgent, weights, total_episodes, total_reward, init = initializeTensor()
        self.assertEqual(total_episodes, 10000)

    def test_initialize_tensor2(self):
        cBandit, myAgent, weights, total_episodes, total_reward, init = initializeTensor()
        self.assertEqual(type(myAgent), ContextualBanditAgent.agent)

    def test_initialize_tensor3(self):
        cBandit, myAgent, weights, total_episodes, total_reward, init = initializeTensor()
        self.assertEqual(type(cBandit), ContextualBandit.contextual_bandit)