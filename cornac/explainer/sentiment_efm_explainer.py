
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
    
    def explain_recommendations(self, recommendations, num_features=3, index = False):
        """"
        recommendations: dataframe, columns name as [user_id, item_id]
        """
        explanations = pd.DataFrame(columns=["user_id", "item_id", "explanations"])
        self.recommendations = recommendations 

        with tqdm(total=self.recommendations.shape[0], desc="Computing explanations: ", position=0, leave=True) as pbar:

            for _, row in self.recommendations.iterrows():
                explanations = pd.concat([explanations,self.explain_one_recommendation_to_user(row.user_id, row.item_id, num_features, index)])
                pbar.update()

        return explanations 
    
    def explain_one_recommendation_to_user(self, user_id, item_id, num_features=3, index=False):
        """
        get aspect with the highest score of the item, and at the same time, being the user's most cared aspect.
        :param user_id:
        :param item_id: 
        :return: a distionary of {"user_id": user_id, "item_id": item_id, "explanations": [{"aspect": aspect, "score": score}, {}, {} ...]}
        """
        self.num_most_cared_aspects = num_features
        
        # if index is not True, then user_id and item_id are real id in dataset, then map it to index
        if not index:
            user_id = self.dataset.uid_map[user_id]
            item_id = self.dataset.iid_map[item_id]
        
        id_aspect_map = {v:k for k, v in self.dataset.sentiment.aspect_id_map.items()}
        predicted_user_aspect_scores = np.dot(self.model.U1[user_id], self.model.V.T)
        predicted_item_aspect_scores = np.dot(self.model.U2[item_id], self.model.V.T)
        
        user_top_cared_aspects_ids = (-predicted_user_aspect_scores).argsort()[:self.num_most_cared_aspects]
        user_top_cared_aspects = [id_aspect_map[aid] for aid in user_top_cared_aspects_ids]
        user_top_cared_aspects_score = predicted_item_aspect_scores[user_top_cared_aspects_ids]
        
        if not index:
            # map back to real id
            user_idx2id = {v: k for k, v in self.dataset.uid_map.items()}
            item_idx2id = {v: k for k, v in self.dataset.iid_map.items()}
            user_id = user_idx2id[user_id]
            item_id = item_idx2id[item_id]
        explanation = {}
        for i, aspect in enumerate(user_top_cared_aspects):
            explanation[aspect]=user_top_cared_aspects_score[i]
        return pd.DataFrame({"user_id": user_id, "item_id": item_id, "explanations": [explanation]})