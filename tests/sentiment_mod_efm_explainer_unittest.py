import unittest
import numpy as np
from cornac.explainer.sentiment_mod_efm_explainer import Mod_EFMExplainer
import cornac
import pandas as pd
from cornac.utils import cache
from cornac.eval_methods import RatioSplit
from cornac.data import Reader, SentimentModality
from cornac.datasets.goodreads import prepare_data
from cornac.models import EFM
from config import cfg
SEED = 42
VERBOSE = False

class TestModEFMExplainer(unittest.TestCase):
    
    @classmethod
    def setUp(cls):
        """
            Prepare data, recommendation model, and explainer for testing
            Purpose: make sure the dataset, recommendation model, and explainer are not None
        """
        
        # Prepare the dataset
        rs = prepare_data(data_name="goodreads", test_size=0, verbose=VERBOSE, sample_size=0.1, dense=True)
        # init the recommendation model
        efm = EFM()
        cls.rec_model = efm.fit(rs.train_set)
        cls.dataset = efm.train_set  
        
        # Init the explainer
        cls.explainer = Mod_EFMExplainer(cls.rec_model, cls.dataset)
        # Make sure the recommender and explainer is not None
        assert cls.rec_model is not None
        assert cls.explainer is not None
        print("TestModEFM Setup complete")
    
    @classmethod
    def test_explain_one_recommendation_to_user(cls):
        """
            Test the function 'explain_one_recommendation_to_user'
                Input: one u-i pair
                Expected Output: a dataframe with columns: user_id, item_id, explanations
            Purpose: make sure the output is a dataframe with expected columns
        """
        # randomly pick a user_id and item_id
        user_id = np.random.choice(list(cls.dataset.user_ids))
        item_id = np.random.choice(list(cls.dataset.item_ids))
        
        feature_k = 3
        threshold = 3.0
        
        # Get explanations
        explanation = cls.explainer.explain_one_recommendation_to_user(user_id=user_id, item_id=item_id, feature_k=feature_k, threshold=threshold)
        # Make sure the number of explanations is equal to the number of most cared aspects
        # make sure the score in explanations is higher than the threshold
        print(f"exfplana: {explanation}")
        if len(explanation)>0:
            for _, value in explanation.items():
                assert int(value) >= threshold
        assert len(explanation) <= feature_k, "The length of explanation should be less than desired number of futures" # can be zero, the Mod_EFMExp may return empty explanation if all scores are less than threshold
        print("TestModEFM test_explain_one_recommendation_to_user complete")
    
    @classmethod    
    def test_explain_recommendations(cls):
        """
            Test the function 'explain_recommendations'
                Input: multiple u-i pairs in dataframe format
                Expected Output: a dataframe with columns: user_id, item_id, explanations
            Purpose: make sure the output is a dataframe with expected columns
        """
        
        # select 5 users
        users = [k for k in cls.dataset.uid_map.keys()][:5]
        len_users = len(users)
        rec_k = 10
        # Get recommendations for the selected users
        recommendations = cls.rec_model.recommend_to_multiple_users(user_ids=users, k=rec_k)
        
        feature_k = 3
        threshold = 0.0
        # Call the method being tested: generate explanations for all u-i pairs in the recommendations
        explanations = cls.explainer.explain_recommendations(
            recommendations, feature_k=feature_k, threshold = threshold
        )
        # check the explanations not None, in right fornmat, and expected length
        assert explanations is not None
        assert len(explanations) <= len_users * rec_k
        assert 'explanations' in set(explanations.columns)
        for index, row in explanations.iterrows():
            if len(row['explanations'])>0:
                for _, value in row['explanations'].items():
                    assert int(value) >= threshold
        print("TestModEFM test_explain_recommendations complete")

if __name__ == '__main__':
    unittest.main()