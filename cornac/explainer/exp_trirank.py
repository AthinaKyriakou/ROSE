import pandas as pd
import numpy as np
from .explainer import Explainer


class Exp_TriRank(Explainer):
    """Explainer from TriRank: Review-aware Explainable Recommendation by Modeling Aspects.
    
    Parameters
    ----------
    rec_model: object, recommender model
        The recommender model to be explained.
    dataset: object, dataset
        The dataset object that is used to explain.
    name: string, optional, default: 'Exp_TriRank'
    
    References
    ----------
    [1]     He, Xiangnan, Tao Chen, Min-Yen Kan, and Xiao Chen. 2014. \
    TriRank: Review-aware Explainable Recommendation by Modeling Aspects. \
    In the 24th ACM international on conference on information and knowledge management (CIKM'15). \
    ACM, New York, NY, USA, 1661-1670. DOI: https://doi.org/10.1145/2806416.2806504
    """

    def __init__(self, rec_model, dataset, name="Exp_TriRank"):
        super().__init__(name=name, rec_model=rec_model, dataset=dataset)
        
        aspect_id_map = self.dataset.sentiment.aspect_id_map
        self.id_to_aspect = {v: k for k, v in aspect_id_map.items()}

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
        
        """
        if self.model is None:
            raise NotImplementedError("The model is None.")
        
        if not hasattr(self.model, "X") or not hasattr(self.model, "Y"):
            raise AttributeError("The explainer does not support this recommender.")
        
        feature_k = kwargs.get("feature_k", 10)

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
        
        item_aspect = self.model.X.getrow(item_idx).toarray().flatten()
        top_k_item_aspect = np.argsort(item_aspect)[-feature_k:] 

        explanation = []
        for aspect in top_k_item_aspect:
            user_interest = self.model.Y.getrow(user_idx).toarray().flatten()[aspect]
            item_aspect_score = item_aspect[aspect]
            aspect_text = self.id_to_aspect[aspect]
            explanation.append((aspect_text, item_aspect_score, user_interest))

        return explanation
