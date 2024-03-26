import unittest
import numpy as np
import pandas as pd
from config import cfg

from cornac.models import MF, EMF, NEMF, ALS
from cornac.datasets.goodreads import prepare_data

class Test_MF_recommenders(unittest.TestCase):
    """
        Test for all Matrix Factorization recommenders
        Using artificial dense data
        Test model.fit and model.recommend
    """
    
    def test_MF(self):
        """
        Test for recommender cornac.models.MF
        """
        rs = prepare_data(data_name="goodreads_uir_1000", test_size=0, verbose=False, sample_size=1, dense=True)
        mf = MF(**cfg.model.mf)
        mf.fit(rs.train_set)
        assert mf is not None
        assert mf.train_set is not None
        
        users = [k for k in rs.train_set.uid_map.keys()][:5]
        len_users = len(users)
        rec_k = 10
        recommendations = mf.recommend(user_ids=users, n=rec_k)
        assert len(recommendations) == len_users * rec_k
        assert isinstance(recommendations, pd.DataFrame)
        assert set(recommendations.columns) == {'user_id', 'item_id', 'prediction'}
        
    def test_EMF(self):
        """
        Test for recommender cornac.models.EMF
        """
        rs = prepare_data(data_name="goodreads_uir_1000", test_size=0, verbose=False, sample_size=1, dense=True)
        emf = EMF(**cfg.model.emf)
        emf.fit(rs.train_set)
        assert emf is not None
        assert emf.train_set is not None
        assert emf.edge_weight_matrix is not None
        assert emf.edge_weight_matrix.shape == (rs.train_set.num_users, rs.train_set.num_items)
        
        users = [k for k in rs.train_set.uid_map.keys()][:5]
        len_users = len(users)
        rec_k = 10
        recommendations = emf.recommend(user_ids=users, n=rec_k)
        assert len(recommendations) == len_users * rec_k
        assert isinstance(recommendations, pd.DataFrame)
        assert set(recommendations.columns) == {'user_id', 'item_id', 'prediction'}
        
    def test_NEMF(self):
        """
        Test for recommender cornac.models.NEMF
        """
        rs = prepare_data(data_name="goodreads_uir_1000", test_size=0, verbose=False, sample_size=1, dense=True)
        nemf = NEMF(**cfg.model.nemf)
        nemf.fit(rs.train_set)
        assert nemf is not None
        assert nemf.train_set is not None
        assert nemf.edge_weight_matrix is not None
        assert nemf.edge_weight_matrix.shape == (rs.train_set.num_users, rs.train_set.num_items)
        assert nemf.novel_matrix is not None
        assert nemf.novel_matrix.shape == (rs.train_set.num_users, rs.train_set.num_items)
        
        users = [k for k in rs.train_set.uid_map.keys()][:5]
        len_users = len(users)
        rec_k = 10
        recommendations = nemf.recommend(user_ids=users, n=rec_k)
        assert len(recommendations) == len_users * rec_k
        assert isinstance(recommendations, pd.DataFrame)
        assert set(recommendations.columns) == {'user_id', 'item_id', 'prediction'}
        
    def test_ALS(self):
        """
        Test for recommender cornac.models.ALS
        """
        rs = prepare_data(data_name="goodreads_uir_1000", test_size=0, verbose=False, sample_size=1, dense=True)
        als = ALS(**cfg.model.als)
        als.fit(rs.train_set)
        assert als is not None
        assert als.train_set is not None
        
        users = [k for k in rs.train_set.uid_map.keys()][:5]
        len_users = len(users)
        rec_k = 10
        recommendations = als.recommend(user_ids=users, n=rec_k)
        assert len(recommendations) == len_users * rec_k
        assert isinstance(recommendations, pd.DataFrame)
        assert set(recommendations.columns) == {'user_id', 'item_id', 'prediction'}
        
        
if __name__ == '__main__':
    unittest.main()