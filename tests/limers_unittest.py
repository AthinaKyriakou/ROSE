import unittest
import os
import sys
import math
import numpy as np
import pandas as pd

from cornac.eval_methods.ratio_split import RatioSplit
from cornac.data.modality import FeatureModality
from cornac.datasets.goodreads import prepare_data
from cornac.models.fm_py import FMRec
from cornac.explainer.limers import LimeRSExplainer
from config import cfg

class TestLIMERS_user_item_features(unittest.TestCase):
    """
        Test for LIMERS
        Using artificial dense data to test LimeRSExplainer generate explanations when both user and item features are used for
            1. explain_one_recommendation_to_user
            2. explain_recommendations
        
        Assume that the recommender is already tested in fm_unittest.py
    """
    @classmethod
    def setUpClass(cls):
        cls.rs = prepare_data(data_name="goodreads_limers", test_size=0.2, dense=True, item=True, user=True, sample_size=0.1, seed=21)
        cls.FMmodel = FMRec()
        cls.FMmodel.fit(cls.rs.train_set)
    
    def test_fm_one_explanation(self):
        """test using fm model, limers can generate one explanation; check length, and check the user and item returned in the explanation match the input"""
        user = self.FMmodel.train_set.pick_top_users(1)[0]
        item = self.FMmodel.train_set.pick_top_items(1)[0]
        explainer = LimeRSExplainer(rec_model=self.FMmodel, dataset=self.FMmodel.train_set)
        explanations = explainer.explain_one_recommendation_to_user(user, item)
        self.assertEqual(len(explanations),1)
        self.assertEqual(explanations.iloc[0,0], user)
        self.assertEqual(explanations.iloc[0,1], item)
    
    def test_fm_multiple_explanations(self):
        """test using fm model, limers can generate multiple explanations; check the output is matches the input"""
        user = self.FMmodel.train_set.pick_top_users(1)[0]
        item = self.FMmodel.train_set.pick_top_items(10)
        rec = pd.DataFrame({"user_id": [user for _ in item], "item_id": item})
        explainer = LimeRSExplainer(rec_model=self.FMmodel, dataset=self.FMmodel.train_set)
        explanations = explainer.explain_recommendations(rec)
        self.assertEqual(len(explanations),10)

class TestLIMERS_item_features(unittest.TestCase):
    """
        Test for LIMERS
        Using artificial dense data to test LimeRSExplainer generate explanations when item features are used for
            1. explain_one_recommendation_to_user
            2. explain_recommendations
        
        Assume that the recommender is already tested in fm_unittest.py
    """
    @classmethod
    def setUpClass(cls):
        cls.rs = prepare_data(data_name="goodreads_limers", test_size=0.2, dense=True, item=True, user=False, sample_size=0.1, seed=21)
        cls.FMmodel = FMRec()
        cls.FMmodel.fit(cls.rs.train_set)
    
    def test_fm_one_explanation(self):
        """test using fm model, limers can generate one explanation; check length, and check the user and item returned in the explanation match the input"""
        user = self.FMmodel.train_set.pick_top_users(1)[0]
        item = self.FMmodel.train_set.pick_top_items(1)[0]
        explainer = LimeRSExplainer(rec_model=self.FMmodel, dataset=self.FMmodel.train_set)
        explanations = explainer.explain_one_recommendation_to_user(user, item)
        self.assertEqual(len(explanations),1)
        self.assertEqual(explanations.iloc[0,0], user)
        self.assertEqual(explanations.iloc[0,1], item)
    
    def test_fm_multiple_explanations(self):
        """test using fm model, limers can generate multiple explanations; check the output is matches the input"""
        user = self.FMmodel.train_set.pick_top_users(1)[0]
        item = self.FMmodel.train_set.pick_top_items(10)
        rec = pd.DataFrame({"user_id": [user for _ in item], "item_id": item})
        explainer = LimeRSExplainer(rec_model=self.FMmodel, dataset=self.FMmodel.train_set)
        explanations = explainer.explain_recommendations(rec)
        self.assertEqual(len(explanations),10)

class TestLIMERS_user_features(unittest.TestCase):
    """
        Test for LIMERS
        Using artificial dense data to test LimeRSExplainer generate explanations when both user features are used for
            1. explain_one_recommendation_to_user
            2. explain_recommendations
        
        Assume that the recommender is already tested in fm_unittest.py
    """
    @classmethod
    def setUpClass(cls):
        cls.rs = prepare_data(data_name="goodreads_limers", test_size=0.2, dense=True, item=False, user=True, sample_size=0.1, seed=21)
        cls.FMmodel = FMRec()
        cls.FMmodel.fit(cls.rs.train_set)
    
    def test_fm_one_explanation(self):
        """test using fm model, limers can generate one explanation; check length, and check the user and item returned in the explanation match the input"""
        user = self.FMmodel.train_set.pick_top_users(1)[0]
        item = self.FMmodel.train_set.pick_top_items(1)[0]
        explainer = LimeRSExplainer(self.FMmodel, self.FMmodel.train_set)
        explanations = explainer.explain_one_recommendation_to_user(user, item)
        self.assertEqual(len(explanations),1)
        self.assertEqual(explanations.iloc[0,0], user)
        self.assertEqual(explanations.iloc[0,1], item)
    
    def test_fm_multiple_explanations(self):
        """test using fm model, limers can generate multiple explanations; check the output is matches the input"""
        user = self.FMmodel.train_set.pick_top_users(1)[0]
        item = self.FMmodel.train_set.pick_top_items(10)
        rec = pd.DataFrame({"user_id": [user for _ in item], "item_id": item})
        explainer = LimeRSExplainer(self.FMmodel, self.FMmodel.train_set)
        explanations = explainer.explain_recommendations(rec)
        self.assertEqual(len(explanations),10)

if __name__ == '__main__':
    unittest.main()


