""" UnitTest """
import unittest
import logging
from relbert import Trainer

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.DEBUG, datefmt='%Y-%m-%d %H:%M:%S')


class Test(unittest.TestCase):
    """ Test """

    def test(self):
        model = Trainer('albert-base-v1', epoch=1, export_dir='tests/ckpt')
        model.train(progress_interval=1)


if __name__ == "__main__":
    unittest.main()
