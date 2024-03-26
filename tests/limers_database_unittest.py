import unittest
import os
import sys
import numpy as np
import pandas as pd

from cornac.eval_methods.ratio_split import RatioSplit
from cornac.data.modality import FeatureModality
from cornac.models.fm_py import FMRec
from config import cfg

class TestLIMERSdataset(unittest.TestCase):
    """
        Test for LIMERS dataset
        Using artificial dense data to test dataset subclass's functions work as expected. 
            1. check dataset has the same column and rows
            2. check features are transformed correctly
            3. check feature transformation function is valid
    """
    @classmethod
    def setUpClass(cls):
        cls.train = pd.read_csv(**cfg.goodreads_limres)
        cls.genres = pd.read_csv(**cfg.goodreads_genres)
        cls.features = np.array([[x,y] for [x,y] in zip(cls.genres['item_id'].to_numpy(), cls.genres['feature'].to_numpy())])
        cls.data_triple = [(u,i,r) for (u,i,r) in zip(cls.train['user_id'].to_numpy(), cls.train['item_id'].to_numpy(), cls.train['rating'].to_numpy())]
        cls.rs = RatioSplit(data=cls.data_triple, seed=24, item_feature = FeatureModality(cls.features))
        cls.dataset = FMRec().fit(cls.rs.train_set).train_set

    def test_data_preparation(self):
        """check if training data has correct columns"""
        expected_name = pd.Series([0,0], index=['user_id', 'item_id']).index
        expected_count = self.rs.train_size
        self.assertCountEqual(self.dataset.training_df.columns, expected_name)
        self.assertEqual(len(self.dataset.training_df),expected_count)
    
    def test_item_features(self):
        """check if item with multiple features values are transformed property"""
        #create feature:item_idx lookup from rs.item_feature
        item_info_short = pd.DataFrame({'item_id':list(map(str,self.rs.item_feature.features[:,0])), 'feature': list(map(str, self.rs.item_feature.features[:,1]))})
        iid_map = pd.DataFrame({'item_id': list(map(str, self.rs.train_set.iid_map.keys())), 'item_indices': list(map(str, self.rs.train_set.iid_map.values()))})
        item_info_short = item_info_short.merge(iid_map, on='item_id', how='left')
        item_info_short = item_info_short.dropna().drop(columns=['item_id'])
        item_info_short.columns = ['feature', 'item_id']
        #find item_idx which has multiple genres and exists in training data. 
        item_multi_genres = item_info_short.groupby(['item_id']).count().reset_index()
        item_multi_genres = self.dataset.training_df.merge(item_multi_genres, on='item_id', how='left').dropna().sort_values('feature', ascending=False)
        #find the item_idx with the most number of genres from rs
        item_idx = item_multi_genres.iloc[0,1]  
        expected_num_features = int(item_multi_genres.iloc[0,2])
        #find the actual number of genres for item_idx in the training df
        actual_num_features = len(self.dataset.item_features[self.dataset.item_features['item_id']==str(item_idx)])
        self.assertEqual(expected_num_features, actual_num_features)
    
    def test_convert_to_feature_long(self):
        """check convert_to_feature_long properly transform feature types to columns"""
        expected_names = self.genres['feature'].unique().tolist()
        expected_names.insert(0, 'item_id')
        expected_names.insert(0, 'user_id')
        actual_names = self.dataset.merge_uir_with_features(self.dataset.training_df).columns.to_list()
        self.assertCountEqual(expected_names, actual_names)



if __name__ == '__main__':
    unittest.main()