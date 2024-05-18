import numpy as np
import pandas as pd
from .explainer import Explainer
from tqdm.auto import tqdm


class Exp_ComparERObj(Explainer):
    """Explainer for ComparERObj (Explainable Recommendation with Comparative Constraints on Product Aspects.)

    Parameters
    ----------
    rec_model: object, recommender model
        The recommender model to be explained.
    dataset: object, dataset
        The dataset object that is used to explain.
    name: string, optional, default: 'Exp_ComparERObj'

    References
    ----------
    [1] Trung-Hoang Le and Hady W. Lauw. "Explainable Recommendation with Comparative Constraints on Product Aspects."
    ACM International Conference on Web Search and Data Mining (WSDM). 2021.

    [2] Yongfeng Zhang, Guokun Lai, Min Zhang, Yi Zhang, Yiqun Liu, and Shaoping Ma. 2014.
    Explicit factor models for explainable recommendation based on phrase-level sentiment analysis.
    https://doi.org/10.1145/2600428.2609579
    
    [3] https://github.com/PreferredAI/tutorials/blob/master/recommender-systems/07_explanations.ipynb

    """

    def __init__(
        self,
        rec_model,
        dataset,
        name="Exp_ComparERObj",
    ):

        super().__init__(name, rec_model, dataset)
        # self.U1 = self.model.U1
        # self.U2 = self.model.U2
        # self.H1 = self.model.H1
        # self.H2 = self.model.H2
        # self.V = self.model.V

        if self.model is None:
            raise NotImplementedError("The model is None.")

    def explain_one_recommendation_to_user(self, user_id, item_id, **kwargs):
        """Get aspect with the highest score of the item, and at the same time, being the user's most cared aspect.

        Parameters
        ----------
        user_id: str
            One user's id.
        item_id: str
            One item's id.
        feature_k: int, optional, default:3
            Number of features in explanations created by explainer.

        Returns
        -------
        explanation: dict
            Explanations as a dictionary of aspect and score.
        """
        # num_features from kwargs
        self.num_most_cared_aspects = kwargs.get("feature_k", 3)
        user_id = self.dataset.uid_map[user_id]
        item_id = self.dataset.iid_map[item_id]

        id_aspect_map = {v: k for k, v in self.dataset.sentiment.aspect_id_map.items()}
        predicted_user_aspect_scores = np.dot(self.model.U1[user_id], self.model.V.T)
        predicted_item_aspect_scores = np.dot(self.model.U2[item_id], self.model.V.T)

        user_top_cared_aspects_ids = (-predicted_user_aspect_scores).argsort()[
            : self.num_most_cared_aspects
        ]
        user_top_cared_aspects = [
            id_aspect_map[aid] for aid in user_top_cared_aspects_ids
        ]
        user_top_cared_aspects_score = predicted_item_aspect_scores[
            user_top_cared_aspects_ids
        ]

        explanation = {}
        for i, aspect in enumerate(user_top_cared_aspects):
            explanation[aspect] = user_top_cared_aspects_score[i]
        return explanation
