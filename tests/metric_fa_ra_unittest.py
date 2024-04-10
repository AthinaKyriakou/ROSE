import unittest
import numpy as np
import pandas as pd
from config import cfg

from cornac.models import EFM, MTER
from cornac.explainer import EFMExplainer, MTERExplainer
from cornac.metrics_explainer import FA, RA
from cornac.datasets.goodreads import prepare_data

class TestFARA(unittest.TestCase):
    """
        Test for FA, RA
        Can work with Explainer-Explainer pairs:
            - EFMExplainer-MTERExplainer
            
        Not work with:
            - LimeRSExplainer: no comparing object
            - EMFExplainer, PHI4MFExplainer: elements in explanations are not in sequential order
        
    """
    def test_FA_RA(self):
        """
        Test for FA & RA with cornac.explainer.EFMExplainer and explainer cornac.explainer.MTERExplainer
        """
        # Prepare the dataset
        print("Start testing FA & RA")
        rs = prepare_data(data_name="goodreads", test_size=0, sample_size=0.1, dense=True)

        efm = EFM()
        efm = efm.fit(rs.train_set)
        efm_exp = EFMExplainer(efm, rs.train_set)
        
        mter = MTER(max_iter=200)
        mter = mter.fit(rs.train_set)
        mter_exp = MTERExplainer(mter, rs.train_set)
        
        users = [k for k in rs.train_set.uid_map.keys()] 
        items = [k for k in rs.train_set.iid_map.keys()] 
        # artifically create user-item pairs
        num_pairs = 10
        feature_k = 10
        user_item = pd.DataFrame({'user_id': users[:num_pairs], 'item_id': items[:num_pairs]})

        exp_1 = efm_exp.explain_recommendations(recommendations=user_item, feature_k=feature_k)
        exp_1['explanations'].apply(lambda x: [v for v in x.keys()])
        exp_1 = exp_1[['user_id', 'item_id', 'explanations']].values
        
        exp_2 = mter_exp.explain_recommendations(recommendations=user_item, feature_k=feature_k)
        exp_2['explanations'].apply(lambda x: [v for v in x.keys()])
        exp_2 = exp_2[['user_id', 'item_id', 'explanations']].values
        
        fa = FA()
        value_fa, list_fa = fa.compute(exp_1, exp_2)
        
        ra = RA()
        value_ra, list_ra = ra.compute(exp_1, exp_2)

        assert value_fa >= 0.0
        assert value_ra >= 0.0
        assert len(list_fa) == len(list_ra)
        
        print("Test FA & RA complete")
        
if __name__ == '__main__':
    unittest.main()