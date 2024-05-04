import unittest
import numpy as np
import pandas as pd
from config import cfg

from cornac.models import MF, EMF, NEMF, ALS
from cornac.explainer import Exp_SU4EMF, Exp_ALS, Exp_PHI4MF
from cornac.datasets.goodreads import prepare_data

class Test_MF_explainers(unittest.TestCase):
    """
        Test for all Matrix Factorization explainers
        Using artificial dense data for EMFExplainer and ALSExplainer
        Using large rating dataset for PHI4MFExplainer, but only use 1%
        
        Recommender-Explainer pairs:
            - ALS-ALSExplainer
            - EMF-EMFExplainer
            - NEMF-EMFExplainer
            - MF-PHI4MFExplainer
            - EMF-PHI4MFExplainer
            - NEMF-PHI4MFExplainer

        Test explainer.explain_recommendations and explainer.explain_one_recommendation_to_user
        
        In this test, we assume the recommender is already tested in mf_recommenders_unittest.py
    """
    def test_ALS_ALSExplainer(self):
        """
        Test for explainer cornac.explainer.ALSExplainer with recommender cornac.models.ALS
        """
        rs = prepare_data(data_name="goodreads_uir_1000", test_size=0, verbose=False, sample_size=1, dense=True)
        als = ALS(**cfg.model.als)
        als.fit(rs.train_set)
        users = [k for k in rs.train_set.uid_map.keys()][:5]
        len_users = len(users)
        rec_k = 10
        recommendations = als.recommend_to_multiple_users(user_ids=users, k=rec_k)
        
        als_exp = Exp_ALS(als, rs.train_set)
        assert als_exp is not None
        
        explanations = als_exp.explain_recommendations(recommendations=recommendations, feature_k=10)
        assert explanations is not None
        assert len(explanations) == len_users * rec_k
        assert isinstance(explanations, pd.DataFrame)
        assert set(explanations.columns) == {'user_id', 'item_id', 'prediction', 'explanations'}
        
        user_id = recommendations['user_id'].values[0]
        item_id = recommendations['item_id'].values[0]
        explanation = als_exp.explain_one_recommendation_to_user(user_id, item_id, feature_k=10)
        assert explanation is not None
        assert isinstance(explanation, dict)
        
    def test_EMF_EMFExplainer(self):
        """
        Test for explainer cornac.explainer.EMFExplainer with recommender cornac.models.EMF
        """
        rs = prepare_data(data_name="goodreads_uir_1000", test_size=0, verbose=False, sample_size=1, dense=True)
        emf = EMF(**cfg.model.emf)
        emf.fit(rs.train_set)
        users = [k for k in rs.train_set.uid_map.keys()][:5]
        len_users = len(users)
        rec_k = 10

        recommendations = emf.recommend_to_multiple_users(user_ids=users, k=rec_k)
        emf_exp = Exp_SU4EMF(emf, rs.train_set)
        assert emf_exp is not None

        explanations = emf_exp.explain_recommendations(recommendations=recommendations, feature_k=10)
        assert explanations is not None
        assert len(explanations) == len_users * rec_k
        assert isinstance(explanations, pd.DataFrame)

        user_id = recommendations['user_id'].values[0]
        item_id = recommendations['item_id'].values[0]
        explanation = emf_exp.explain_one_recommendation_to_user(user_id, item_id)
        assert explanation is not None
        assert isinstance(explanation, dict)
            
    def test_NEMF_EMFExplainer(self):
        """
        Test for explainer cornac.explainer.EMFExplainer with recommender cornac.models.NEMF
        """
        rs = prepare_data(data_name="goodreads_uir_1000", test_size=0, verbose=False, sample_size=1, dense=True)
        nemf = NEMF(**cfg.model.nemf)
        nemf.fit(rs.train_set)
        users = [k for k in rs.train_set.uid_map.keys()][:5]
        len_users = len(users)
        rec_k = 10

        recommendations = nemf.recommend_to_multiple_users(user_ids=users, k=rec_k)
        emf_exp = Exp_SU4EMF(nemf, rs.train_set)
        assert emf_exp is not None

        explanations = emf_exp.explain_recommendations(recommendations=recommendations, feature_k=10)
        assert explanations is not None
        assert len(explanations) == len_users * rec_k
        assert isinstance(explanations, pd.DataFrame)

        user_id = recommendations['user_id'].values[0]
        item_id = recommendations['item_id'].values[0]
        explanation = emf_exp.explain_one_recommendation_to_user(user_id, item_id)
        assert explanation is not None
        assert isinstance(explanation, dict)
        
    def test_MF_PHI4MFExplainer(self):
        """
        Test for explainer cornac.explainer.PHI4MFExplainer with recommender cornac.models.MF
        """
        rs = prepare_data(data_name="goodreads_uir", test_size=0, verbose=False, sample_size=0.01)
        mf = MF(**cfg.model.mf)
        mf.fit(rs.train_set)
        users = [k for k in rs.train_set.uid_map.keys()][:5]
        len_users = len(users)
        rec_k = 10

        recommendations = mf.recommend_to_multiple_users(user_ids=users, k=rec_k)
        phi_exp = Exp_PHI4MF(mf, rs.train_set)
        assert phi_exp is not None

        explanations = phi_exp.explain_recommendations(recommendations=recommendations, feature_k=10)
        assert explanations is not None
        assert len(explanations) == len_users * rec_k
        assert isinstance(explanations, pd.DataFrame)

        user_id = recommendations['user_id'].values[0]
        item_id = recommendations['item_id'].values[0]
        explanation = phi_exp.explain_one_recommendation_to_user(user_id, item_id, feature_k=10)
        assert explanation is not None
        assert isinstance(explanation, list)
        
    def test_EMF_PHI4MFExplainer(self):
        """
        Test for explainer cornac.explainer.PHI4MFExplainer with recommender cornac.models.EMF
        """
        rs = prepare_data(data_name="goodreads_uir", test_size=0, verbose=False, sample_size=0.01)
        emf = EMF(**cfg.model.emf)
        emf.fit(rs.train_set)
        users = [k for k in rs.train_set.uid_map.keys()][:5]
        len_users = len(users)
        rec_k = 10

        recommendations = emf.recommend_to_multiple_users(user_ids=users, k=rec_k)
        phi_exp = Exp_PHI4MF(emf, rs.train_set)
        assert phi_exp is not None

        explanations = phi_exp.explain_recommendations(recommendations=recommendations, feature_k=10)
        assert explanations is not None
        assert len(explanations) == len_users * rec_k
        assert isinstance(explanations, pd.DataFrame)

        user_id = recommendations['user_id'].values[0]
        item_id = recommendations['item_id'].values[0]
        explanation = phi_exp.explain_one_recommendation_to_user(user_id, item_id, feature_k=10)
        assert explanation is not None
        assert isinstance(explanation, list)
        
    def test_NEMF_PHI4MFExplainer(self):
        """
        Test for explainer cornac.explainer.PHI4MFExplainer with recommender cornac.models.NEMF
        """
        rs = prepare_data(data_name="goodreads_uir", test_size=0, verbose=False, sample_size=0.01)
        nemf = NEMF(**cfg.model.nemf)
        nemf.fit(rs.train_set)
        users = [k for k in rs.train_set.uid_map.keys()][:5]
        len_users = len(users)
        rec_k = 10

        recommendations = nemf.recommend_to_multiple_users(user_ids=users, k=rec_k)
        phi_exp = Exp_PHI4MF(nemf, rs.train_set)
        assert phi_exp is not None

        explanations = phi_exp.explain_recommendations(recommendations=recommendations, feature_k=10)
        assert explanations is not None
        assert len(explanations) == len_users * rec_k
        assert isinstance(explanations, pd.DataFrame)

        user_id = recommendations['user_id'].values[0]
        item_id = recommendations['item_id'].values[0]
        explanation = phi_exp.explain_one_recommendation_to_user(user_id, item_id, feature_k=10)
        assert explanation is not None
        assert isinstance(explanation, list)
        
    def test_not_valid(self):
        """
        Test for not valid pairs:
            - ALS-EMFExplainer
            - ALS-PHI4MFExplainer
            - MF-ALSExplainer
            - MF-EMFExplainer
        Other un-valid pairs are similar to these pairs
        """
        als = ALS(**cfg.model.als)
        mf = MF(**cfg.model.mf)

        rs = prepare_data(data_name="goodreads_uir_1000", test_size=0, verbose=False, sample_size=1, dense=True)
        
        als.fit(rs.train_set)
        users = [k for k in rs.train_set.uid_map.keys()][:5]
        als_rec = als.recommend_to_multiple_users(user_ids=users, k=10)
        
        mf.fit(rs.train_set)
        mf_rec = mf.recommend_to_multiple_users(user_ids=users, k=10)
        
        with self.assertRaises(AttributeError):
            exp = Exp_SU4EMF(als, rs.train_set)
            exp.explain_recommendations(als_rec, feature_k=10)
        with self.assertRaises(AttributeError):
            exp = Exp_PHI4MF(als, rs.train_set)
            exp.explain_recommendations(als_rec, feature_k=10)
        with self.assertRaises(AttributeError):
            exp = Exp_ALS(mf, rs.train_set)
            exp.explain_recommendations(mf_rec, feature_k=10)
        with self.assertRaises(AttributeError):
            exp = Exp_SU4EMF(mf, rs.train_set)
            exp.explain_recommendations(mf_rec, feature_k=10)
            
        
if __name__ == '__main__':
    unittest.main()