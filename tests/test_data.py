""" UnitTest """
import unittest
import logging
from relbert import get_training_data

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.DEBUG, datefmt='%Y-%m-%d %H:%M:%S')


class Test(unittest.TestCase):
    """ Test """

    def test(self):
        positive, negative, _ = get_training_data('semeval2012')
        print(positive)


if __name__ == "__main__":
    unittest.main()
