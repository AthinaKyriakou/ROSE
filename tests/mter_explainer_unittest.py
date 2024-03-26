import unittest
import numpy as np
from cornac.explainer.sentiment_mter_explainer import MTERExplainer
import cornac
import pandas as pd
from cornac.utils import cache
from cornac.eval_methods import RatioSplit
from cornac.data import Reader, SentimentModality
from cornac.models import MTER
from cornac.datasets.goodreads import prepare_data
from config import cfg
SEED = 12
VERBOSE = True

class TestMTERExplainer(unittest.TestCase):
    @classmethod
    def setUp(cls):
        """
            Prepare data, recommendation model, and explainer for testing
            Purpose: make sure the dataset, recommendation model, and explainer are not None
        """
        # Prepare the dataset
        rs = prepare_data(data_name="goodreads", test_size=0, verbose=VERBOSE, sample_size=0.1, dense=True)
        
        # init the recommendation model
        mter = MTER(**cfg.model.mter)
        cls.rec_model = mter.fit(rs.train_set)
        cls.dataset = mter.train_set  
        
        # Init the explainer
        cls.explainer = MTERExplainer(cls.rec_model, cls.dataset)
        # Make sure the recommender and explainer is not None
        assert cls.rec_model is not None
        assert cls.explainer is not None
        print("TestMTER Setup complete")
        
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

        num_top_aspects = 3
        num_top_opinions = 3
        explanation = cls.explainer.explain_one_recommendation_to_user(user_id, item_id, num_top_aspects, num_top_opinions)
        
        # Make sure the output is corresponding to the input user_id and item_id
        # Perform assertions to verify the expected behavior
        assert explanation["user_id"][0] == user_id
        assert explanation["item_id"][0] == item_id
        assert len(explanation["explanations"][0]) == num_top_aspects
        assert len(list(explanation["explanations"][0].values())[0]) == num_top_opinions

        print("TestMTER test_explain_one_recommendation_to_user complete")
    
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
        recommendations = cls.rec_model.recommend(user_ids=users, n=rec_k)
        
        num_top_aspects = 3
        num_top_opinions = 3
        explanations = cls.explainer.explain_recommendations(recommendations, num_features = num_top_aspects, num_top_opinions=num_top_opinions)

        # check the explanations not None, in right fornmat, and expected length
        assert explanations is not None
        assert len(explanations) == len_users * rec_k
        # Ensure that the user_id and item_id match the input recommendations
        assert isinstance(explanations, pd.DataFrame)
        assert set(explanations.columns) == {'user_id', 'item_id', 'explanations'}

if __name__ == '__main__':
    unittest.main()