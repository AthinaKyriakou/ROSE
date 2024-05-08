import pandas as pd
import numpy as np
from .explainer import Explainer


class Exp_EMF(Explainer):
    """Explainer from Explainable Matrix Factorization for Collaborative Filtering. Explains by W matrix in the paper.
    
    Parameters
    ----------
    rec_model: object, recommender model
        The recommender model to be explained.
    dataset: object, dataset
        The dataset object that is used to explain.
    name: string, optional, default: 'Exp_EMF'
    
    References
    ----------
    [1] B. Abdollahi and O. Nasraoui, “Explainable Matrix Factorization for Collaborative Filtering,” \
    ACM Press, 2016, pp. 5-6. doi: 10.1145/2872518.2889405.
    """

    def __init__(self, rec_model, dataset, name="Exp_EMF"):
        super().__init__(name=name, rec_model=rec_model, dataset=dataset)

    def explain_one_recommendation_to_user(self, user_id, item_id, **kwargs):
        """Provide explanation for one user and one item

        Parameters
        ----------
        user_id: str
            One user's id.
        item_id: str
            One item's id.
        feature_k: int, optional, default:10
            Number of features in explanations created by explainer.

        Returns
        -------
        explanations: float
            The W matrix value of the user and item.
        """
        if self.model is None:
            raise NotImplementedError("The model is None.")
        if not hasattr(self.model, "edge_weight_matrix"):
            raise AttributeError("The explainer does not support this recommender.")
        if self.model.edge_weight_matrix is None:
            raise NotImplementedError("The model is not trained yet.")

        uir_df = pd.DataFrame(
            np.array(self.dataset.uir_tuple).T, columns=["user", "item", "rating"]
        )
        uir_df["user"] = uir_df["user"].astype(int)
        uir_df["item"] = uir_df["item"].astype(int)
        if user_id not in self.dataset.uid_map:
            return []
        if item_id not in self.dataset.iid_map:
            return []
        user_idx = self.dataset.uid_map[user_id]
        item_idx = self.dataset.iid_map[item_id]
        
        W = self.model.edge_weight_matrix
        explanation = W[user_idx, item_idx]

        return explanation
