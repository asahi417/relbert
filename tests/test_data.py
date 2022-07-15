""" UnitTest """
import unittest
import logging
from relbert.data import get_training_data, get_analogy_data

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.DEBUG, datefmt='%Y-%m-%d %H:%M:%S')


class Test(unittest.TestCase):
    """ Test """

    def test(self):
        get_analogy_data()


if __name__ == "__main__":
    unittest.main()
