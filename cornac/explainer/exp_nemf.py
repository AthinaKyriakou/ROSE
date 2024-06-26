import pandas as pd
import numpy as np
from .explainer import Explainer


class Exp_NEMF(Explainer):
    """Explainer for Personalised novel and explainable matrix factorisation. Explains by E matrix in the paper.
    
    Parameters
    ----------
    rec_model: object, recommender model
        The recommender model to be explained.
    dataset: object, dataset
        The dataset object that is used to explain.
    name: string, optional, default: 'Exp_NEMF'
    
    References
    ----------
    [1] L. Coba, P. Symeonidis, and M. Zanker, “Personalised novel and explainable matrix factorisation,” 
    Data & Knowledge Engineering, vol. 122, pp. 142-158, Jul. 2019, doi: 10.1016/j.datak.2019.06.003.
    """

    def __init__(self, rec_model, dataset, name="Exp_NEMF"):
        super().__init__(name=name, rec_model=rec_model, dataset=dataset)

    def explain_one_recommendation_to_user(self, user_id, item_id, **kwargs):
        """Provide explanation for one user and one item

        Parameters
        ----------
        user_id: str
            One user's id.
        item_id: str
            One item's id.

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
            return 0
        if item_id not in self.dataset.iid_map:
            return 0
        user_idx = self.dataset.uid_map[user_id]
        item_idx = self.dataset.iid_map[item_id]
        if self.model.is_unknown_user(user_idx):
            return 0
        if self.model.is_unknown_item(item_idx):
            return 0
        
        W = self.model.edge_weight_matrix
        explanation = W[user_idx, item_idx]

        return explanation
