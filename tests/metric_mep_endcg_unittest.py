import unittest
import numpy as np
import pandas as pd
from config import cfg

from cornac.models import EMF, NEMF, MF
from cornac.metrics_explainer import Metric_Exp_MEP as MEP
from cornac.metrics_explainer import Metric_Exp_EnDCG as EnDCG
from cornac.datasets.goodreads import prepare_data

class TestMEP_EnDCG(unittest.TestCase):
    """
    Test metrics MEP and EnDCG
    Using artificial dense data
    metrics MEP and EnDCG can work with model that has edge_weight_matrix, as EMF and NEMF
    assume that the recommender is already tested in mf_recommenders_unittest.py
    
    Test metrics.compute
    """
    def test_MEP_EMF(self):
        """
        Test for metric cornac.metrics_explainer.MEP with recommender cornac.models.EMF
        """
        rs = prepare_data(data_name="goodreads_uir_1000", test_size=0, verbose=False, sample_size=1, dense=True)
        emf = EMF(**cfg.model.emf)
        emf.fit(rs.train_set)
        mep = MEP()
        value, distribution = mep.compute(emf)
        
        assert value >= 0.0
        assert len(distribution) == rs.train_set.num_users
        
    def test_EnDCG_EMF(self):
        """
        Test for metric cornac.metrics_explainer.EnDCG with recommender cornac.models.EMF
        """
        rs = prepare_data(data_name="goodreads_uir_1000", test_size=0, verbose=False, sample_size=1, dense=True)
        emf = EMF(**cfg.model.emf)
        emf.fit(rs.train_set)
        endcg = EnDCG()
        value, distribution = endcg.compute(emf)
        
        assert value >= 0.0
        assert len(distribution) == rs.train_set.num_users
        
    def test_MEP_NEMF(self):
        """
        Test for metric cornac.metrics_explainer.MEP with recommender cornac.models.NEMF
        """
        rs = prepare_data(data_name="goodreads_uir_1000", test_size=0, verbose=False, sample_size=1, dense=True)
        nemf = NEMF(**cfg.model.nemf)
        nemf.fit(rs.train_set)
        mep = MEP()
        value, distribution = mep.compute(nemf)
        
        assert value >= 0.0
        assert len(distribution) == rs.train_set.num_users
        
    def test_EnDCG_NEMF(self):
        """
        Test for metric cornac.metrics_explainer.EnDCG with recommender cornac.models.NEMF
        """
        rs = prepare_data(data_name="goodreads_uir_1000", test_size=0, verbose=False, sample_size=1, dense=True)
        nemf = NEMF(**cfg.model.nemf)
        nemf.fit(rs.train_set)
        endcg = EnDCG()
        value, distribution = endcg.compute(nemf)
        
        assert value >= 0.0
        assert len(distribution) == rs.train_set.num_users
            
    
    def test_not_valid_MEP(self):
        """
        Test for metric cornac.metrics_explainer.MEP with not valid recommender (cornac.models.MF as example)
        """
        rs = prepare_data(data_name="goodreads_uir_1000", test_size=0, verbose=False, sample_size=1, dense=True)
        mf = MF(**cfg.model.mf)
        mf.fit(rs.train_set)
        mep = MEP()
        with self.assertRaises(NotImplementedError):
            value, distribution = mep.compute(mf)
        
    def test_not_valid_EnDCG(self):
        """
        Test for metric cornac.metrics_explainer.EnDCG with not valid recommender (cornac.models.MF as example)
        """
        rs = prepare_data(data_name="goodreads_uir_1000", test_size=0, verbose=False, sample_size=1, dense=True)
        mf = MF(**cfg.model.mf)
        mf.fit(rs.train_set)
        endcg = EnDCG()
        with self.assertRaises(NotImplementedError):
            value, distribution = endcg.compute(mf)
            
    def test_not_train_MEP(self):
        """
        Test for metric cornac.metrics_explainer.MEP with not trained recommender (cornac.models.EMF as example)
        """
        emf = EMF(**cfg.model.emf)
        mep = MEP()
        
        with self.assertRaises(AttributeError):
            value, distribution = mep.compute(emf)
        
    def test_not_train_EnDCG(self):
        """
        Test for metric cornac.metrics_explainer.EnDCG with not trained recommender (cornac.models.EMF as example)
        """
        emf = EMF(**cfg.model.emf)
        endcg = EnDCG()
        
        with self.assertRaises(AttributeError):
            value, distribution = endcg.compute(emf)
        
        
if __name__ == '__main__':
    unittest.main()