import unittest
import numpy as np
from cornac.explainer.sentiment_efm_explainer import EFMExplainer
import cornac
import pandas as pd
from cornac.utils import cache
from cornac.eval_methods import RatioSplit
from cornac.data import Reader, SentimentModality
from cornac.datasets.goodreads import prepare_data
from cornac.models import EFM
from config import cfg
SEED = 12
VERBOSE = True

class TestEFMExplainer(unittest.TestCase):
    
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
        cls.explainer = EFMExplainer(cls.rec_model, cls.dataset)
        # Make sure the recommender and explainer is not None
        assert cls.rec_model is not None
        assert cls.explainer is not None
        print("TestEFM Setup complete")
    
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
        
        num_features = 3
        index  = False
        # Get explanations
        explanation = cls.explainer.explain_one_recommendation_to_user(user_id=user_id, item_id=item_id, num_features=num_features)
        # Make sure the number of explanations is equal to the number of most cared aspects
        assert len(explanation) == num_features, "The length of explanation should be the specified number of futures"
        print("TestEFM test_explain_one_recommendation_to_user complete")
    
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
        
        num_features = 3
        # Call the method being tested: generate explanations for all u-i pairs in the recommendations
        explanations = cls.explainer.explain_recommendations(
            recommendations, num_features=num_features
        )
        # check the explanations not None, in right fornmat, and expected length
        assert explanations is not None
        assert len(explanations) == len_users * rec_k
        assert isinstance(explanations, pd.DataFrame)
        assert 'explanations' in set(explanations.columns)
        print("TestEFM test_explain_recommendations complete")

if __name__ == '__main__':
    unittest.main()