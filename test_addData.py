from unittest import TestCase
from addData import *

class Test(TestCase):
    def test_get_integer1(self):
        numberOfTrials = get_integer(2)
        self.assertEqual(numberOfTrials, 2)

    def test_get_integer2(self):
        with self.assertRaises(SystemExit) as test:
            get_integer("A")
        the_exception = test.exception
        self.assertEqual(type(the_exception), type(SystemExit(1)))

    def test_get_integer3(self):
        with self.assertRaises(SystemExit) as test:
            get_integer("3.5")
        the_exception = test.exception
        self.assertEqual(type(the_exception), type(SystemExit(1)))

    def test_get_integer4(self):
        numberOfTrials = get_integer("8")
        self.assertEqual(numberOfTrials, 8)

    def test_get_integer5(self):
        numberOfTrials= get_integer(3.5)
        self.assertEqual(numberOfTrials, 3)
