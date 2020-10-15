from unittest import TestCase
from OneTrial import *

class TestRandomSeedNumber(TestCase):
    def test_get_integer(self):
        seed = RandomSeedNumber
        test = seed.get_integer(1)
        self.assertEqual(test, 1)

