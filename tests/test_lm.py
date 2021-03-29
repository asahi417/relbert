""" UnitTest """
import unittest
import logging
import torch
from relbert import RelBERT, get_semeval_data

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.DEBUG, datefmt='%Y-%m-%d %H:%M:%S')


class Test(unittest.TestCase):
    """ Test """

    def test_lm(self):
        all_positive, all_negative, all_relation_type = get_semeval_data()

        model = RelBERT('albert-base-v1')
        iterator = model.preprocess(
            positive_samples=all_positive,
            negative_sample=all_negative,
            pairwise_input=True,
            parallel=False
        )

        data_loader = torch.utils.data.DataLoader(
            iterator, batch_size=2, shuffle=False, drop_last=False)

        logging.debug('\t* run LM inference')
        for encode in data_loader:
            print(encode)
            input()


if __name__ == "__main__":
    unittest.main()
