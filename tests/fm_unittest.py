import unittest
import os
import sys
import math
import numpy as np
import pandas as pd
from config import cfg


from cornac.eval_methods.ratio_split import RatioSplit
from cornac.data.modality import FeatureModality
from cornac.datasets.goodreads import prepare_data
from cornac.models.fm_py.recom_fm_py import FMRec

class TestFMModel_user_item_features(unittest.TestCase):
    """
    Using artificial dense data to test FM model when both user and item features are used
        1. test fm.recommend
        2. test fm.score for one [user, item]
        3. test fm.score for one user
    """
    @classmethod
    def setUpClass(cls):
        """test using item and user features, fm is able to train, predict and recommend"""
        cls.rs = prepare_data(data_name="goodreads_limers", test_size=0.2, dense=True, item=True, user=True, sample_size=0.5, seed=21)
        cls.model = FMRec()
        cls.model.fit(cls.rs.train_set)
 
    def test_fm_recommendations(self):
        """check prediction data type and contains no nan values;
            correct count of recommendations are provided to user;
            check the same user ids are used in the recommendations as input"""
        user = self.model.train_set.pick_top_users(2)
        recommendations = self.model.recommend_to_multiple_users(user,k=10)
        result = recommendations['user_id'].unique()
        self.assertEqual(20, len(recommendations))
        self.assertEqual(recommendations['prediction'].dtype, 'float64')
        self.assertEqual(recommendations['prediction'].notna().all(), True)
        self.assertCountEqual(result, user) 

    def test_fm_one_prediction(self):
        """check prediction output has no nan values
            check the correct count of predictions are provided to user"""
        user = self.model.train_set.pick_top_users(1)
        items = self.model.train_set.pick_top_items(1)
        user_idx = [str(self.model.train_set.uid_map[x]) for x in user]
        items_idx = [str(self.model.train_set.iid_map[x]) for x in items]
        #df = pd.DataFrame({"user_id": [user_idx[0] for _ in range(len(items_idx))], "item_id": items_idx})
        result = self.model.score(user_idx[0], items_idx[0])
        self.assertEqual(isinstance(result, float), True)
    
    def test_fm_user_predictions(self):
        """check prediction output has no nan values
            check the correct count of predictions are provided to user"""
        user = self.model.train_set.pick_top_users(1)
        user_idx = [str(self.model.train_set.uid_map[x]) for x in user]
        result = self.model.score(user_idx[0])
        self.assertEqual(len(self.model.train_set.iid_map), len(result))
        self.assertEqual(any(math.isnan(x) for x in result), False)

class TestFMModel_item_features(unittest.TestCase):
    """
    Using artificial dense data to test FM model when item features are used
        1. test fm.recommend
        2. test fm.score for one [user, item]
        3. test fm.score for one user
        4. test predictions can be given to unknown user
        5. test recommendations can be given to unknown user
        6. test when training df contain item_id that is not in the item_feature input, an exception can be raised
    """
    @classmethod
    def setUpClass(cls):
        """test using item features, fm is able to train, predict and recommend"""
        cls.rs = prepare_data(data_name="goodreads_limers", test_size=0.2, dense=True, item=True, user=False, sample_size=0.5, seed=21)
        cls.model = FMRec()
        cls.model.fit(cls.rs.train_set)
 
    def test_fm_recommendations(self):
        """check prediction data type and contains no nan values;
            correct count of recommendations are provided to user;
            check the same user ids are used in the recommendations as input"""
        user = self.model.train_set.pick_top_users(2)
        recommendations = self.model.recommend_to_multiple_users(user,k=10)
        result = recommendations['user_id'].unique()
        self.assertEqual(20, len(recommendations))
        self.assertEqual(recommendations['prediction'].dtype, 'float64')
        self.assertEqual(recommendations['prediction'].notna().all(), True)
        self.assertCountEqual(result, user) 

    def test_fm_one_prediction(self):
        """check prediction output has no nan values
            check the correct count of predictions are provided to user"""
        user = self.model.train_set.pick_top_users(1)
        items = self.model.train_set.pick_top_items(1)
        user_idx = [str(self.model.train_set.uid_map[x]) for x in user]
        items_idx = [str(self.model.train_set.iid_map[x]) for x in items]
        #df = pd.DataFrame({"user_id": [user_idx[0] for _ in range(len(items_idx))], "item_id": items_idx})
        result = self.model.score(user_idx[0], items_idx[0])
        self.assertEqual(isinstance(result, float), True)
    
    def test_fm_user_predictions(self):
        """check prediction output has no nan values
            check the correct count of predictions are provided to user"""
        user = self.model.train_set.pick_top_users(1)
        user_idx = [str(self.model.train_set.uid_map[x]) for x in user]
        result = self.model.score(user_idx[0])
        self.assertEqual(len(self.model.train_set.iid_map), len(result))
        self.assertEqual(any(math.isnan(x) for x in result), False)

    def test_fm_predictions_unknown_user(self):
        """ For an unknown user but known items during training,
            check valid prediction can be given
            check prediction output has no nan values"""
        user = self.model.train_set.pick_top_users(1) #use unmapped user_id as 'unknown user'
        items = self.model.train_set.pick_top_items(3)
        user_idx = self.model.train_set.uid_map[user[0]]
        items_idx = [str(self.model.train_set.iid_map[x]) for x in items]
        result = self.model.score(user_idx)
        self.assertEqual(any(math.isnan(x) for x in result), False)
    
    def test_fm_recommendations_unknown_user(self):
        """Check recommendations can be given to unknown user
           check predictions are valid"""
        user = self.model.train_set.pick_top_users(1) #use unmapped user_id as 'unknown user'
        recommendations = self.model.recommend_to_multiple_users(user, k=10)
        self.assertEqual(recommendations['prediction'].dtype, 'float64')
        self.assertEqual(recommendations['prediction'].notna().all(), True)

    def test_fm_test_exception(self):
        "Check to ensure when training df contain item_id that is not in the item_feature input, an exception can be raised"
        train_exception = pd.read_csv(**cfg.goodreads_limres_exception)
        genres = pd.read_csv(**cfg.goodreads_genres)
        features = np.array([[x,y] for [x,y] in zip(genres['item_id'].to_numpy(), genres['feature'].to_numpy())])
        data_triple = [(u,i,r) for (u,i,r) in zip(train_exception['user_id'].to_numpy(), train_exception['item_id'].to_numpy(), train_exception['rating'].to_numpy())]
        rs = RatioSplit(data=data_triple, seed=24, item_feature = FeatureModality(features))
        model_exception = FMRec()
        with self.assertRaises(ValueError):
            model_exception.fit(rs.train_set)
            raise ValueError("training data contain items which features are unknown")

class TestFMModel_user_features(unittest.TestCase):
    """
    Using artificial dense data to test FM model when user features are used
        1. test fm.recommend
        2. test fm.score for one [user, item]
        3. test fm.score for one user
        4. test predictions can be given to unknown item
        5. test recommendations can be given to unknown item
        6. test when training df contain user_id that is not in the user_feature input, an exception can be raised
    """
    @classmethod
    def setUpClass(cls):
        """test using user features, fm is able to train, predict and recommend"""
        cls.rs = prepare_data(data_name="goodreads_limers", test_size=0.2, dense=True, item=False, user=True, sample_size=0.5, seed=21)
        cls.model = FMRec()
        cls.model.fit(cls.rs.train_set)
 
    def test_fm_recommendations(self):
        """check prediction data type and contains no nan values;
            correct count of recommendations are provided to user;
            check the same user ids are used in the recommendations as input"""
        user = self.model.train_set.pick_top_users(2)
        recommendations = self.model.recommend_to_multiple_users(user,k=10)
        result = recommendations['user_id'].unique()
        self.assertEqual(20, len(recommendations))
        self.assertEqual(recommendations['prediction'].dtype, 'float64')
        self.assertEqual(recommendations['prediction'].notna().all(), True)
        self.assertCountEqual(result, user) 

    def test_fm_one_prediction(self):
        """check prediction output has no nan values
            check the correct count of predictions are provided to user"""
        user = self.model.train_set.pick_top_users(1)
        items = self.model.train_set.pick_top_items(1)
        user_idx = [str(self.model.train_set.uid_map[x]) for x in user]
        items_idx = [str(self.model.train_set.iid_map[x]) for x in items]
        #df = pd.DataFrame({"user_id": [user_idx[0] for _ in range(len(items_idx))], "item_id": items_idx})
        result = self.model.score(user_idx[0], items_idx[0])
        self.assertEqual(isinstance(result, float), True)
    
    def test_fm_user_predictions(self):
        """check prediction output has no nan values
            check the correct count of predictions are provided to user"""
        user = self.model.train_set.pick_top_users(1)
        user_idx = [str(self.model.train_set.uid_map[x]) for x in user]
        result = self.model.score(user_idx[0])
        self.assertEqual(len(self.model.train_set.iid_map), len(result))
        self.assertEqual(any(math.isnan(x) for x in result), False)

    def test_fm_predictions_unknown_item(self):
        """ For an unknown user but known items during training,
            check valid prediction can be given
            check prediction output has no nan values"""
        user = self.model.train_set.pick_top_users(1) #use unmapped user_id as 'unknown user'
        items = self.model.train_set.pick_top_items(3)
        user_idx = [str(self.model.train_set.uid_map[x]) for x in user]
        items_idx = self.model.train_set.iid_map[items[0]]
        result = self.model.score(items_idx)
        self.assertEqual(any(math.isnan(x) for x in result), False)
    
    def test_fm_test_exception(self):
        "Check to ensure when training df contain user_id that is not in the user_feature input, an exception can be raised"
        train_exception = pd.read_csv(**cfg.goodreads_limres_exception)
        uid_aspects = pd.read_csv(**cfg.goodreads_uid_aspects)
        features = np.array([[x,y] for [x,y] in zip(uid_aspects['user_id'].to_numpy(), uid_aspects['feature'].to_numpy())])
        data_triple = [(u,i,r) for (u,i,r) in zip(train_exception['user_id'].to_numpy(), train_exception['item_id'].to_numpy(), train_exception['rating'].to_numpy())]
        rs = RatioSplit(data=data_triple, seed=24, user_feature = FeatureModality(features))
        model_exception = FMRec()
        with self.assertRaises(ValueError):
            model_exception.fit(rs.train_set)
            raise ValueError("training data contain users which features are unknown")

if __name__ == '__main__':
    unittest.main()