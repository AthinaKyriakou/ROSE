import pandas as pd
import numpy as np
from .explainer import Explainer

class EMFExplainer(Explainer):
    """Explainer from Explainable Matrix Factorization for Collaborative Filtering

    Parameters
    ----------
    rec_model: object, recommender model
        The recommender model to be explained.
    dataset: object, dataset
        The dataset object that is used to explain.
    name: string, optional, default: 'EMF'
    
    References
    ----------
    B. Abdollahi and O. Nasraoui, “Explainable Matrix Factorization for Collaborative Filtering,” \
    ACM Press, 2016, pp. 5-6. doi: 10.1145/2872518.2889405.
    Code Reference
    ----------
    https://github.com/ludovikcoba/recoxplainer/blob/master/recoxplainer/explain/model_based_emf.py
    """
    
    def __init__(self, rec_model, dataset, name='EMF'):
        super().__init__(name=name, rec_model=rec_model, dataset=dataset) 

    def explain_one_recommendation_to_user(self, user_id, item_id, **kwargs):
        """
        provide explanation for one user and one item
        user_id: one user
        item_id: one item
        return: a dictionary containing mapping from rating score to count
        """
        if self.model is None:
            raise NotImplementedError("The model is None.")
        if not hasattr(self.model, 'sim_users'):
            raise AttributeError("The explainer does not support this recommender.")
        if self.model.sim_users is None:
            raise NotImplementedError("The model is not trained yet.")
                
        uir_df = pd.DataFrame(np.array(self.dataset.uir_tuple).T, columns=['user', 'item', 'rating'])
        uir_df['user'] = uir_df['user'].astype(int)
        uir_df['item'] = uir_df['item'].astype(int)
        if user_id not in self.dataset.uid_map:
            return []
        if item_id not in self.dataset.iid_map:
            return []
        user_idx = self.dataset.uid_map[user_id]
        item_idx = self.dataset.iid_map[item_id]
        item_all_user_rating = uir_df[uir_df['item'] == item_idx]
        sim_user_list = self.model.sim_users[user_idx]
        item_sim_user_rating = item_all_user_rating[item_all_user_rating['user'].isin(sim_user_list)]
        explain_df = item_sim_user_rating.groupby('rating').count()
        explanation = {}
        for idx, row in explain_df.iterrows():
            explanation[idx] = row['user']

        return explanation
