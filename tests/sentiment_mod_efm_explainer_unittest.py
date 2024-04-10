import unittest
import numpy as np
from cornac.explainer.sentiment_mod_efm_explainer import Mod_EFMExplainer
import cornac
import pandas as pd
from cornac.utils import cache
from cornac.eval_methods import RatioSplit
from cornac.data import Reader, SentimentModality
from cornac.models import EFM
from config import cfg
SEED = 42
VERBOSE = False

class TestEFMExplainer(unittest.TestCase):
    
    @classmethod
    def setUp(cls):
        # Create a mock recommendation model and dataset for testing
        sentiment = Reader().read(**cfg.goodreads_sentiment)
        # Load rating and sentiment information
        rating = Reader(min_item_freq=20).read(**cfg.goodreads_rating)
        sentiment_modality = SentimentModality(data=sentiment)

        rs = RatioSplit(
            data=rating,
            test_size=0.2,
            exclude_unknowns=True,
            sentiment=sentiment_modality,
            verbose=VERBOSE,
            seed=SEED,
        )
        efm = EFM()
        cls.rec_model = efm.fit(rs.train_set)
        cls.dataset = efm.train_set  
        
        # Create an instance of the EFMExplainer class
        cls.explainer = Mod_EFMExplainer(cls.rec_model, cls.dataset)
        print("TestModEFM Setup complete")
    
    @classmethod
    def test_explain_one_recommendation_to_user(cls):
        user_id = 1  
        item_id = 2 
        index  = True
        # Call the method being tested
        explanation = cls.explainer.explain_one_recommendation_to_user(user_id=user_id, item_id=item_id, feature_k=3, index=index)
        
        # Perform assertions to verify the expected behavior
        assert explanation["user_id"][0] == user_id
        assert explanation["item_id"][0] == item_id

        assert isinstance(explanation["aspect"][0], str) # "aspect should be of type str"
        assert isinstance(explanation["aspect_score"][0], float) # "aspect_score should be of type float"
        assert isinstance(explanation["max_aspect_name"][0], str) # "aspect should be of type str"
        assert isinstance(explanation["max_aspect_score"][0], float) # "aspect_score should be of type float"
        print("TestModEFM test_explain_one_recommendation_to_user complete")
    
    @classmethod    
    def test_explain_recommendations(cls):
        recommendations = pd.DataFrame({
            "user_id": [1, 2, 3],
            "item_id": [100, 200, 300]
        })
        feature_k = 3
        index = True

        explanations = cls.explainer.explain_recommendations(
            recommendations, feature_k, index
        )

        # Perform assertions to validate the output
        assert len(explanations) == recommendations.shape[0]
        assert all(col in explanations.columns for col in ["user_id", "item_id", "aspect", "aspect_score"])  
        print("TestModEFM test_explain_recommendations complete")

if __name__ == '__main__':
    unittest.main()