import unittest
import numpy as np
import pandas as pd
from config import cfg

from cornac.models import EMF, ALS, EFM
from cornac.explainer import EMFExplainer, ALSExplainer, PHI4MFExplainer, EFMExplainer
from cornac.metrics_explainer import PGF
from cornac.datasets.goodreads import prepare_data

class TestPGF(unittest.TestCase):
    """
        Test for PGF
        Using artificial dense data for EMFExplainer and ALSExplainer
        Using large rating dataset for PHI4MFExplainer, but only use 1%
        
        Can work with Recommender-Explainer pairs:
            - ALS-ALSExplainer
            - EMF-EMFExplainer
            - NEMF-EMFExplainer
            - MF-PHI4MFExplainer
            - EMF-PHI4MFExplainer
            - NEMF-PHI4MFExplainer
        But in this test, we only use:
            - ALS-ALSExplainer
            - EMF-EMFExplainer
            - EMF-PHI4MFExplainer
        Because the other pairs are similar to these pairs
        
        Assume that the recommender is already tested in mf_recommenders_unittest.py
        And the explainer is already tested in mf_explainers_unittest.py
        
        Test PGF.compute
    """
    def test_PGF_EMF_EMF(self):
        """
        Test for PGF with recommender cornac.models.EMF and explainer cornac.explainer.EMFExplainer
        """
        rs = prepare_data(data_name="goodreads_uir_1000", test_size=0, verbose=False, sample_size=1, dense=True)
        emf = EMF(**cfg.model.emf)
        emf.fit(rs.train_set)
        emf_exp = EMFExplainer(emf, rs.train_set)
        users = [k for k in rs.train_set.uid_map.keys()] 
        rec_df = emf.recommend(users, n=10)
        exp = emf_exp.explain_recommendations(recommendations=rec_df, num_features=10)[['user_id', 'item_id', 'explanations']].values
        
        pgf = PGF(phi=10)
        value, distribution = pgf.compute(emf, emf_exp, exp)

        assert value >= 0.0
        assert len(distribution) == rs.train_set.num_users
            
    def test_PGF_ALS_ALS(self):
        """
        Test for PGF with recommender cornac.models.ALS and explainer cornac.explainer.ALSExplainer
        """
        rs = prepare_data(data_name="goodreads_uir_1000", test_size=0, verbose=False, sample_size=1, dense=True)
        als = ALS(**cfg.model.als)
        als.fit(rs.train_set)
        als_exp = ALSExplainer(als, rs.train_set)
        users = [k for k in rs.train_set.uid_map.keys()]
        rec_df = als.recommend(users, n=10)
        exp = als_exp.explain_recommendations(recommendations=rec_df, num_features=10)[['user_id', 'item_id', 'explanations']]
        exp['explanations'] = exp['explanations'].apply(lambda x: [v for v in x.keys()])
        exp = exp[['user_id', 'item_id', 'explanations']].values
        
        pgf = PGF(phi=10)
        value, distribution = pgf.compute(als, als_exp, exp)
        
        assert value >= 0.0
        assert len(distribution) == rs.train_set.num_users
        
    def test_PGF_EMF_PHI4MF(self):
        """
        Test for PGF with recommender cornac.models.EMF and explainer cornac.explainer.PHI4MFExplainer
        PHI4MF only use with sparse dataset, so we use the large dataset, but it takes a lot of time to run, so only use 1% of the dataset
        """
        rs = prepare_data(data_name="goodreads_uir", test_size=0.1, verbose=False, sample_size=0.01)
        emf = EMF(**cfg.model.emf)
        emf.fit(rs.train_set)
        phi_exp = PHI4MFExplainer(emf, rs.train_set)
        users = [k for k in rs.train_set.uid_map.keys()] 
        rec_df = emf.recommend(users, n=10)
        exp = phi_exp.explain_recommendations(recommendations=rec_df, num_features=10)[['user_id', 'item_id', 'explanations']].values
        
        pgf = PGF(phi=10)
        value, distribution = pgf.compute(emf, phi_exp, exp)

        assert value >= 0.0
        assert len(distribution) == rs.train_set.num_users
        
    def test_not_valid_PGF(self):
        """
        Test for metric cornac.metrics_explainer.PGF with not valid explainer (EFMExplainer as example)
        """
        rs_sent = prepare_data(data_name="goodreads", test_size=0, dense=False, item=True, user=True, sample_size=0.01, seed=21)
        efm = EFM(max_iter=200)
        efm.fit(rs_sent.train_set)
        efm_exp = EFMExplainer(rec_model=efm, dataset=rs_sent.train_set)
        
        pgf = PGF(phi=10)
        with self.assertRaises(NotImplementedError):
            pgf.compute(efm, efm_exp, None)
            
    def test_not_train_PGF(self):
        """
        Test for metric cornac.metrics_explainer.PGF with not trained recommender (EMF as example)
        """
        emf = EMF(**cfg.model.emf)
        pgf = PGF(phi=10)
        with self.assertRaises(AttributeError):
            pgf.compute(emf, None, None)
        
if __name__ == '__main__':
    unittest.main()
