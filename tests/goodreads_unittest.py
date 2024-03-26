import unittest
import numpy as np
import pandas as pd
from config import cfg

from cornac.data import Dataset, SentimentModality, FeatureModality
from cornac.datasets.goodreads import prepare_data

class TestGoodreads(unittest.TestCase):
    """
        Test for goodreads dataset
        Using artificial dense data to test by calling 'data_name'. Also check using invalid 'data_name' would generate none.
        
    """
    def test_rs_for_limers(self):
        rs = prepare_data(data_name="goodreads_limers", test_size=0.2, dense=True, item=True, user=True, sample_size=0.1, seed=21)
        self.assertIsInstance(rs.test_set, Dataset)
        self.assertIsInstance(rs.train_set, Dataset)
        self.assertIsInstance(rs.train_set.item_feature, FeatureModality)
        self.assertIsInstance(rs.train_set.user_feature, FeatureModality)
    
    def test_rs_for_sentiment(self):
        rs = prepare_data(data_name="goodreads", test_size=0.2, dense=True, item=True, user=True, sample_size=0.1, seed=21)
        self.assertIsInstance(rs.test_set, Dataset)
        self.assertIsInstance(rs.train_set, Dataset)
        self.assertIsInstance(rs.train_set.sentiment, SentimentModality)

    def test_rs_for_mf(self):
        rs = prepare_data(data_name="goodreads_uir_1000", test_size=0.2, dense=True, item=True, user=True, sample_size=0.1, seed=21)
        self.assertIsInstance(rs.test_set, Dataset)
        self.assertIsInstance(rs.train_set, Dataset)  
        self.assertGreaterEqual(len(rs.train_set.uir_tuple), 0)
    
    def test_invalid_dataset(self):
        try:
            rs = prepare_data(data_name="my_favoriate_dataset", test_size=0.2, dense=True, item=True, user=True, sample_size=0.1, seed=21)
        except None:
            assert True


if __name__ == '__main__':
    unittest.main()