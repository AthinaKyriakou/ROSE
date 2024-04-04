
import numpy as np
import pandas as pd
from .explainer import Explainer
from tqdm.auto import tqdm
        
class MTERExplainer(Explainer):
    def __init__(self, rec_model, dataset, name="MTER_Exp", ):
        super().__init__(name, rec_model, dataset)
        self.G1 = self.model.G1
        self.G2 = self.model.G2
        self.G3 = self.model.G3
        self.U = self.model.U
        self.I = self.model.I
        self.A = self.model.A
        self.O = self.model.O
        
        if self.model is None:
            raise NotImplementedError("The model is None.")
        if self.model.name not in ['MTER']:
            raise AttributeError("The explainer does not support this recommender.")
        
        
    def explain_one_recommendation_to_user(self, user_id, item_id, **kwargs):
        """
        get aspect performs best and user's opinion word of that aspect.
        :param user_id:
        :param item_id: 
        ==== does this necessary
        :num_top_aspects: number of aspects used as explanation
        :num_top_opinions: number of opinions used to explain each aspect
        ====
        :return: a distionary of {aspect: [{opinion: score}, {opinion: score}, ...}], aspect: [{opinion: score}, {opinion: score}, ...]}
        """
        # num_features=3, num_top_opinions=3
        self.num_top_aspects = kwargs.get("num_features", 3)
        self.num_top_opinions = kwargs.get("num_top_opinions", 3)

        user_id = self.dataset.uid_map[user_id]
        item_id = self.dataset.iid_map[item_id]


        id_aspect_map = {v:k for k, v in self.dataset.sentiment.aspect_id_map.items()}
        id_opinion_map = {v:k for k, v  in self.dataset.sentiment.opinion_id_map.items()}
        item_id_list = np.array([k for k in self.dataset.sentiment.item_sentiment.keys()])
        idx_list = np.array([k for k in self.dataset.sentiment.sentiment.keys()])
        if item_id not in item_id_list:
            idx_selected = np.random.choice(idx_list, 3, replace=False)
            item_aspect_ids = np.array(list(set([
                tup[0]
                for idx in idx_selected
                for tup in self.dataset.sentiment.sentiment[idx]
            ])))
            item_opinion_ids = np.array(list(set([
                tup[1]
                for idx in idx_selected
                for tup in self.dataset.sentiment.sentiment[idx]
            ])))
        else:
            item_aspect_ids = np.array(list(set([
                tup[0]
                for idx in self.dataset.sentiment.item_sentiment[item_id].values()
                for tup in self.dataset.sentiment.sentiment[idx]
            ])))
            item_opinion_ids = np.array(list(set([
                tup[1]
                for idx in self.dataset.sentiment.item_sentiment[item_id].values()
                for tup in self.dataset.sentiment.sentiment[idx]
            ])))

        #item_aspects = [id_aspect_map[idx] for idx in item_aspect_ids]
        
        ts1 = np.einsum("abc, a->bc", self.model.G1, self.model.U[user_id])
        ts2 = np.einsum("bc, b->c", ts1, self.model.I[item_id])
        predicted_aspect_scores = np.einsum("c, Mc->M", ts2, self.model.A)
        top_aspect_ids = item_aspect_ids[(predicted_aspect_scores[item_aspect_ids]).argsort()[:self.num_top_aspects]]
        top_aspects = [id_aspect_map[idx] for idx in top_aspect_ids]
        explanations = {}
        for top_aspect_id, top_aspect in zip(top_aspect_ids, top_aspects):
            ts1_G2 = np.einsum("abc, a -> bc", self.model.G2, self.model.U[user_id])
            ts2_G2 = np.einsum("bc, b -> c", ts1_G2, self.model.A[top_aspect_id])
            predicted_user_aspect_opinion_scores = np.einsum("c, Mc -> M", ts2_G2, self.model.O)
            
            ts1_G3 = np.einsum("abc, a -> bc", self.model.G3, self.model.I[item_id])
            ts2_G3 = np.einsum("bc, b -> c", ts1_G3, self.model.A[top_aspect_id])
            predicted_item_aspect_opinion_scores = np.einsum("c, Mc -> M", ts2_G3, self.model.O)
            
            predicted_aspect_opinion_scores = np.multiply(predicted_user_aspect_opinion_scores, predicted_item_aspect_opinion_scores)
            top_opinion_ids = item_opinion_ids[(-predicted_aspect_opinion_scores[item_opinion_ids]).argsort()[:self.num_top_opinions]]
            
            top_opinions_scores = {}
            for top_opinion_id in top_opinion_ids:
                top_opinions_scores[id_opinion_map[top_opinion_id]]=predicted_aspect_opinion_scores[top_opinion_id]
            
            explanations[top_aspect]=top_opinions_scores
            
        return explanations