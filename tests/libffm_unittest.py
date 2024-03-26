import unittest
import os
import sys
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from cornac.utils.libffm_mod import LibffmModConverter

class TestLibffm_mod(unittest.TestCase):
    def setUp(self):
        self.user_item_rating = pd.DataFrame({
            'user_id':['100','200','200'],
            'item_id': ['1','2','1'],
            'rating':[4.5,5,4.8]
        })

        self.item_features = pd.DataFrame({
            'item_id':['2','1','1'],
            'production_year': ['2001','1990', '1990'],
            'genres': ['fiction','drama','action']
        })

        self.df = pd.merge(self.user_item_rating, self.item_features, on='item_id')

    def test_libffm_transform(self):
        """check that the output df is the same size as uir data. 
            check transformed data is as expected"""
        converter = LibffmModConverter().fit(self.df, col_rating='rating')
        actual_df = converter.transform(self.df)
        expected_df = pd.DataFrame({
            'rating': [4.5,4.8,5.0],
            'user_id': ['1:1:1', '1:2:1', '1:2:1'],
            'item_id': ['2:3:1','2:3:1','2:4:1'],
            ('production_year', '1990'): ['3:5:1', '3:5:1','3:5:0'],
            ('production_year', '2001'): ['3:6:0', '3:6:0', '3:6:1'],
            ('genres', 'drama'): ['4:7:1',"4:7:1", "4:7:0"],
            ('genres', 'action'): ['4:8:1',"4:8:1","4:8:0"],
            ('genres', 'fiction'): ['4:9:0','4:9:0','4:9:1']
            })
        self.assertEqual(len(actual_df), len(self.user_item_rating))
        assert_frame_equal(actual_df, expected_df)


if __name__ == '__main__':
    unittest.main()

