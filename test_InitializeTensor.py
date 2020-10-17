from unittest import TestCase
from InitializeTensor import *
import ContextualBanditAgent
import ContextualBandit


## @package test_InitializeTensor
## Documentation for test_InitializeTensor.py
# We test the initializeTensor function to make sure that the types of Contextual Bandit and Agent are correct/consistent.

class Test(TestCase):
    ## Documentation for test_initialize_tensor1 function
    # to make sure that total_episodes is not returned as string and expectedly 10,000
    def test_initialize_tensor1(self):
        cBandit, myAgent, weights, total_episodes, total_reward, init = initializeTensor()
        self.assertEqual(total_episodes, 10000)

    ## Documentation for test_initialize_tensor2 function
    #  Once initalizeTensor returns values, the type of ContextualBanditAgent is expected to be matched.
    def test_initialize_tensor2(self):
        cBandit, myAgent, weights, total_episodes, total_reward, init = initializeTensor()
        self.assertEqual(type(myAgent), ContextualBanditAgent.agent)

    ## Documentation for test_initialize_tensor3 function
    #  Once initalizeTensor returns values, the type of ContextualBandit is expected to be matched.
    def test_initialize_tensor3(self):
        cBandit, myAgent, weights, total_episodes, total_reward, init = initializeTensor()
        self.assertEqual(type(cBandit), ContextualBandit.contextual_bandit)
