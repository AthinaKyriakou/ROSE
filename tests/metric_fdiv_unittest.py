import unittest
import numpy as np
import pandas as pd
from config import cfg

from cornac.models import ALS, EFM 
from cornac.explainer import Exp_ALS, Exp_EFM, Exp_EFM
from cornac.metrics_explainer import Metric_Exp_DIV as DIV
from cornac.datasets.goodreads import prepare_data

class TestDIV(unittest.TestCase):
    """
        Test for DIV
        Can work with Recommender-Explainer pairs:
            - FMRec-LimeRSExplainer
            - EFM-EFMExplainer
            - MTER-MTERExplainer
            
            - ALS-ALSExplainer
            - EMF-EMFExplainer
            - NEMF-EMFExplainer
            
        Not work with:
            - MF-PHI4MFExplainer
            - EMF-PHI4MFExplainer
            - NEMF-PHI4MFExplainer
        In this test, we use EFM-EFMExplainer and FMRec-LimeRSExplainer as an example:

    """
    def test_DIV_EFM(self):
        """
        Test for DIV with recommender cornac.models.EFM and explainer cornac.explainer.EFMExplainer
        DIV can work on MTER-MTERExplainer similar to EFM-EFMExplainer
        """
        # Prepare the dataset
        print("Start testing DIV")
        rs = prepare_data(data_name="goodreads", test_size=0, sample_size=0.1, dense=True)
        # init the recommendation model
        efm = EFM()
        efm = efm.fit(rs.train_set)
        # Init the explainer
        efm_exp = Exp_EFM(efm, rs.train_set)

        users = [k for k in rs.train_set.uid_map.keys()] 
        rec_df = efm.recommend_to_multiple_users(users, k=10)
        exp = efm_exp.explain_recommendations(recommendations=rec_df, feature_k=10)[['user_id', 'item_id', 'explanations']].values
        div = DIV()
        value, distribution = div.compute(exp)

        assert value >= 0.0
        # any two explanations have a diversity value
        print("len(distribution) = ", len(distribution))
        print("len(exp) = ", len(exp))
        assert len(distribution) == len(exp)*(len(exp)-1)/2
        
    def test_DIV_FMRec_LIMERS(self):
        """
        Test for DIV with recommender cornac.models.ALS and explainer cornac.explainer.ALSExplainer
        """
        rs = prepare_data(data_name="goodreads_uir_1000", test_size=0, verbose=False, sample_size=1, dense=True)
        als = ALS(**cfg.model.als)
        als.fit(rs.train_set)
        als_exp = Exp_ALS(als, rs.train_set)
        users = [k for k in rs.train_set.uid_map.keys()]
        rec_df = als.recommend_to_multiple_users(users, k=10)
        exp = als_exp.explain_recommendations(recommendations=rec_df, feature_k=10)[['user_id', 'item_id', 'explanations']]
        exp['explanations'].apply(lambda x: [v for v in x.keys()])
        exp = exp[['user_id', 'item_id', 'explanations']].values
        
        div = DIV()
        value, distribution = div.compute(exp)
        
        assert value >= 0.0
        assert len(distribution) == len(exp)*(len(exp)-1)/2
            
        
if __name__ == '__main__':
    unittest.main()