
import numpy as np
import pandas as pd
from .explainer import Explainer
from tqdm.auto import tqdm

class EFMExplainer(Explainer):
    def __init__(self, rec_model, dataset, name="EFM_Exp",):
        super().__init__(name, rec_model, dataset)
        self.U1 = self.model.U1
        self.U2 = self.model.U2
        self.H1 = self.model.H1
        self.H2 = self.model.H2
        self.V = self.model.V
        
        if self.model is None:
            raise NotImplementedError("The model is None.")
        if self.model.name not in ['EFM']:
            raise AttributeError("The explainer does not support this recommender.")
    
    
    def explain_one_recommendation_to_user(self, user_id, item_id, **kwargs):
        """
        get aspect with the highest score of the item, and at the same time, being the user's most cared aspect.
        user_id: one user
        item_id: one item
        feature_k: default 10, number of features in explanations created by explainer
        :return: a distionary of {"user_id": user_id, "item_id": item_id, "explanations": [{"aspect": aspect, "score": score}, {}, {} ...]}
        """
        # num_features from kwargs
        self.num_most_cared_aspects = kwargs.get("feature_k", 3)
        user_id = self.dataset.uid_map[user_id]
        item_id = self.dataset.iid_map[item_id]

        
        id_aspect_map = {v:k for k, v in self.dataset.sentiment.aspect_id_map.items()}
        predicted_user_aspect_scores = np.dot(self.model.U1[user_id], self.model.V.T)
        predicted_item_aspect_scores = np.dot(self.model.U2[item_id], self.model.V.T)
        
        user_top_cared_aspects_ids = (-predicted_user_aspect_scores).argsort()[:self.num_most_cared_aspects]
        user_top_cared_aspects = [id_aspect_map[aid] for aid in user_top_cared_aspects_ids]
        user_top_cared_aspects_score = predicted_item_aspect_scores[user_top_cared_aspects_ids]
        
        explanation = {}
        for i, aspect in enumerate(user_top_cared_aspects):
            explanation[aspect]=user_top_cared_aspects_score[i]
        return explanation