
import numpy as np
import pandas as pd
from .explainer import Explainer
from tqdm.auto import tqdm

class Mod_EFMExplainer(Explainer):
    def __init__(self, rec_model, dataset, name='MEFM'):
        super().__init__(name, rec_model, dataset)
        self.U1 = self.model.U1
        self.U2 = self.model.U2
        self.H1 = self.model.H1
        self.H2 = self.model.H2
        self.V = self.model.V
    
    def explain_recommendations(self, recommendations, num_features=3, index = False):
        """"
        recommendations: dataframe, columns name as [user_id, item_id]
        """
        explanations = pd.DataFrame(columns=["user_id", "item_id", "recommend", "aspect", "aspect_score", "max_aspect_name", "max_aspect_score"])
        self.recommendations = recommendations 

        with tqdm(total=self.recommendations.shape[0], desc="Computing explanations: ", position=0, leave=True) as pbar:

            for _, row in self.recommendations.iterrows():
                explanations = pd.concat([explanations,self.explain_one_recommendation_to_user(row.user_id, row.item_id, num_features, index)])
                pbar.update()

        return explanations 
    
    def explain_one_recommendation_to_user(self, user_id, item_id, num_features=3, index=False, threshold = 3.0):
        """
        get aspect with the highest score of the item, and at the same time, being the user's most cared aspect.
        :param user_id:
        :param item_id: 
        :return: a distionary of {"aspect_name": "aspect_score_of_item"}
        """
        self.num_features = num_features
        self.threshold = threshold
        
        # if index is not True, then user_id and item_id are real id in dataset, then map it to index
        if not index:
            user_id = self.dataset.uid_map[user_id]
            item_id = self.dataset.iid_map[item_id]
        
        id_aspect_map = {v:k for k, v in self.dataset.sentiment.aspect_id_map.items()}
        
        predicted_user_aspect_scores_all = np.dot(self.U1, self.V.T)
        predicted_item_aspect_scores_all = np.dot(self.U2, self.V.T)
        
        predicted_user_aspect_scores = self.standardize_columns(predicted_user_aspect_scores_all)[user_id]
        predicted_item_aspect_scores = self.standardize_columns(predicted_item_aspect_scores_all)[item_id]
        
        user_top_cared_aspects_ids = (-predicted_user_aspect_scores).argsort()[:num_features]
        user_top_cared_aspects = [id_aspect_map[aid] for aid in user_top_cared_aspects_ids]
        
        best_aspect = user_top_cared_aspects[predicted_item_aspect_scores[user_top_cared_aspects_ids].argmax()]
        best_aspect_score = predicted_item_aspect_scores[user_top_cared_aspects_ids].max().astype(float)
        
        recommend = True if best_aspect_score >= self.threshold else False
        
        if not index:
            # map back to real id
            user_idx2id = {v: k for k, v in self.dataset.uid_map.items()}
            item_idx2id = {v: k for k, v in self.dataset.iid_map.items()}
            user_id = user_idx2id[user_id]
            item_id = item_idx2id[item_id]
            
        explanation = pd.DataFrame({
            "user_id": [user_id],
            "item_id": [item_id],
            "recommend": recommend, 
            "aspect": [best_aspect],
            "aspect_score": [best_aspect_score],
            "max_aspect_name": [id_aspect_map[predicted_item_aspect_scores.argmax()]],
            "max_aspect_score": [predicted_item_aspect_scores.max().astype(float)]
        })
        
        return explanation
    
    def standardize_columns(sel, matrix):
        # Compute the column-wise mean and standard deviation
        ## it's normalizing the columns, not standardizing 
        max_val = 5.0
        min_val = 0.0
        column_min = np.min(matrix, axis=0)
        column_max = np.max(matrix, axis=0)

        # Normalize the values of each column based on the minimum and maximum
        normalized_matrix = (matrix - column_min) / (column_max - column_min)

        # Scale the normalized values to the desired range
        scaled_matrix = normalized_matrix * (max_val - min_val) + min_val
        
        return scaled_matrix