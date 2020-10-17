from unittest import TestCase
from addData import *


## @package test_addData
## Documentation for test_addData.py
# The test is for ensuring that the function handle exceptions properly.
# Since we use the number from command line to determine the number of trials, so we could not have non-integer values as the input.
# There are five tests that are expected to pass.
class Test(TestCase):
    ## Documentation for test_get_integer1 function
    #  the input is 2 and it is expected to return as 2.
    def test_get_integer1(self):
        numberOfTrials = get_integer(2)
        self.assertEqual(numberOfTrials, 2)

    ## Documentation for test_get_integer2 function
    #  the input is "A" and it is expected to return SystemExit(1).
    def test_get_integer2(self):
        with self.assertRaises(SystemExit) as test:
            get_integer("A")
        the_exception = test.exception
        self.assertEqual(type(the_exception), type(SystemExit(1)))

    ## Documentation for test_get_integer3 function
    #  the input is "3.5" and it is expected to return SystemExit(1).
    def test_get_integer3(self):
        with self.assertRaises(SystemExit) as test:
            get_integer("3.5")
        the_exception = test.exception
        self.assertEqual(type(the_exception), type(SystemExit(1)))

    ## Documentation for test_get_integer4 function
    #  the input is "8" and it is expected to return 8.
    def test_get_integer4(self):
        numberOfTrials = get_integer("8")
        self.assertEqual(numberOfTrials, 8)

    ## Documentation for test_get_integer5 function
    #  the input is 8.5 and it is expected to return 8.
    def test_get_integer5(self):
        numberOfTrials = get_integer(8.5)
        self.assertEqual(numberOfTrials, 8)
