import pandas as pd
import numpy as np
from .explainer import Explainer

class ALSExplainer(Explainer):
    """Alternating Least Squares of Matrix Factorization for implicit feedback datasets.

    Parameters
    ----------
    rec_model: object, recommender model
        The recommender model to be explained.
    dataset: object, dataset
        The dataset object that is used to explain.
    name: string, optional, default: 'ALS'
    References
    ----------
    Y. Hu, Y. Koren, and C. Volinsky, “Collaborative Filtering for Implicit Feedback Datasets,” in 2008 Eighth IEEE International Conference on Data Mining, Pisa, Italy: IEEE, Dec. 2008, pp. 263-272. doi: 10.1109/ICDM.2008.22.
    
    Code Reference
    ----------
    https://github.com/ludovikcoba/recoxplainer/blob/master/recoxplainer/explain/model_based_als_explain.py
    """
    
    def __init__(self, rec_model, dataset, name="ALS"):
        super().__init__(name=name, rec_model=rec_model, dataset=dataset) 

    def explain_one_recommendation_to_user(self, user_id, item_id, num_features=10):
        """
        provide explanation for one user and one item
        user_id: one user
        item_id: one item
        num_features: number of features to be returned
        return: explanations as a dictionary containing items and their contributions
        """
        self.number_of_contributions = num_features
        uir_df = pd.DataFrame(np.array(self.dataset.uir_tuple).T, columns=['user', 'item', 'rating'])
        if user_id not in self.dataset.uid_map:
            return []
        if item_id not in self.dataset.iid_map:
            return []
        uir_df['user'] = uir_df['user'].astype(int)
        uir_df['item'] = uir_df['item'].astype(int)
        user_idx = self.dataset.uid_map[user_id]
        item_idx = self.dataset.iid_map[item_id]
                
        users_by_item = uir_df[uir_df['item'] == item_idx]['user']
        items_by_user = uir_df[uir_df['user'] == user_idx]['item']

        
        # alpha = 1.0
        # if hasattr(self.model, 'alpha'):
        #     alpha = self.model.alpha
        
        if not hasattr(self.model, 'alpha'):
            raise AttributeError("The explainer does not support this recommender.")
        if self.model is None:
            raise NotImplementedError("The model is None.")
        alpha = self.model.alpha    
        current_interactions = np.zeros(self.num_items)
        user_items = self.dataset.matrix * alpha
        # current_interactions[items_by_user] = user_items[user_idx, items_by_user]
        for item in items_by_user:
            # c_ui = 1 + a r_ui
            current_interactions[item] = user_items[user_idx, item] + 1
        
        C_u = np.diag(current_interactions)
        
        Y_T = self.model.i_factors.T
        Y = self.model.i_factors
        
        W_u = np.matmul(Y_T, np.matmul(C_u, Y)) + self.model.lambda_reg * np.eye(self.model.k)
        
        if len(items_by_user) > 1:
            W_u = np.linalg.inv(W_u)
        else:
            W_u = np.linalg.pinv(W_u)
        
        temp = np.matmul(Y, W_u)
        y_j = Y[item_idx]
        # sim_to_rec = np.matmul(temp, y_j.T)
        sim_to_rec = np.dot(temp, y_j)
        sim_to_rec = sim_to_rec[items_by_user]
        
        confidence = current_interactions[items_by_user]
        
        contribution = {"item": items_by_user, "contribution": sim_to_rec * confidence}
        contribution = pd.DataFrame(contribution)
        contribution = contribution.sort_values(by=['contribution'], ascending=False)
        
        item_idx2id = {v: k for k, v in self.dataset.iid_map.items()}
        contribution['item'] = contribution['item'].apply(lambda x: item_idx2id[x])
        # return {"item_id": contribution['item'].values[:self.number_of_contributions],
        #         "contribution": contribution['contribution'].values[:self.number_of_contributions]}
        
        return {v:k for v, k in zip(contribution['item'].values[:self.number_of_contributions],
                                    contribution['contribution'].values[:self.number_of_contributions])}
