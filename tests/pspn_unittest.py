import unittest
import numpy as np
import pandas as pd
from config import cfg

from cornac.models import FMRec, EFM, EMF
from cornac.explainer import LimeRSExplainer, EFMExplainer, EMFExplainer
from cornac.metrics_explainer import PSPNFNS
from cornac.datasets.goodreads import prepare_data

class TestPSPNModel(unittest.TestCase):
    """
        Test for PSPN metric
        Using artificial dense data to test PSPN produce reasonable results for limers and sentiment explainers. 
        In addition, error is raised when unsupported explainers are passed
        
        Assume that the explainers have already been tested in fm_unittest.py
    """
    def test_evaluate_limers(self):
        """test for (fm, limers), evaluation result is between 0 and 1"""
        rs = prepare_data(data_name="goodreads_limers", test_size=0.2, dense=True, item=True, user=True, sample_size=0.1, seed=21)
        fm = FMRec().fit(rs.train_set)
        limers = LimeRSExplainer(rec_model=fm, dataset=rs.train_set)
        lst_users = [key for key in fm.train_set.uid_map.keys()]
        recommendations = fm.recommend_to_multiple_users(lst_users, k=10)
        explanations = limers.explain_recommendations(recommendations)[['user_id', 'item_id', 'explanations']]
        explanations = explanations[explanations['explanations'] != {}]
        explanations = explanations[['user_id', 'item_id', 'explanations']].values
        pspn = PSPNFNS()
        (ps,pn,fns), _ = pspn.compute(fm, limers, explanations)
        self.assertGreaterEqual(ps, 0)
        self.assertGreaterEqual(pn, 0)
        self.assertGreaterEqual(fns, 0)
        self.assertLessEqual(ps, 1)
        self.assertLessEqual(pn, 1)
        self.assertLessEqual(fns, 1)

    def test_evaluate_sentiment(self):
        """test for (efm, efm_exp), evaluation result is between 0 and 1"""
        rs = prepare_data(data_name="goodreads", test_size=0.2, dense=True, item=True, user=True, sample_size=0.2, seed=21)
        efm = EFM(max_iter=200, verbose=False, seed=6).fit(rs.train_set)
        efm_exp = EFMExplainer(rec_model=efm, dataset=rs.train_set)
        lst_users = [key for key in rs.train_set.uid_map.keys()]
        recommendations = efm.recommend_to_multiple_users(lst_users, k=10)
        explanations = efm_exp.explain_recommendations(recommendations)[['user_id', 'item_id', 'explanations']].values
        pspn = PSPNFNS()
        (ps,pn,fns), _ = pspn.compute(efm, efm_exp, explanations)
        self.assertGreaterEqual(ps, 0)
        self.assertGreaterEqual(pn, 0)
        self.assertGreaterEqual(fns, 0)
        self.assertLessEqual(ps, 1)
        self.assertLessEqual(pn, 1)
        self.assertLessEqual(fns, 1)
    
    def test_unsupported_explainer(self):
        """test the metric would raise value error when unsupported explainers are used"""
        rs = prepare_data(data_name="goodreads_uir_1000", test_size=0.2, dense=True, item=True, user=True, sample_size=0.6, seed=21)
        emf = EMF().fit(rs.train_set)
        emf_exp = EMFExplainer(rec_model=emf, dataset=rs.train_set)
        lst_users = [key for key in rs.train_set.uid_map.keys()]
        recommendations = emf.recommend_to_multiple_users(lst_users, k=10)
        explanations = emf_exp.explain_recommendations(recommendations)[['user_id', 'item_id', 'explanations']].values      
        pspn = PSPNFNS()
        with self.assertRaises(ValueError):
            pspn.compute(emf, emf_exp, explanations)
            raise ValueError(f"Metric {pspn.name} does not support {emf_exp.name}")

if __name__ == '__main__':
    unittest.main()