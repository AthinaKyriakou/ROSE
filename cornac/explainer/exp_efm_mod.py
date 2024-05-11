import numpy as np
import pandas as pd
from .explainer import Explainer
from tqdm.auto import tqdm


class Exp_EFM_Mod(Explainer):
    """Explainer for EFM (Explicit Factor Model) with filter.

    Parameters
    ----------
    rec_model: object, recommender model
        The recommender model to be explained.
    dataset: object, dataset
        The dataset object that is used to explain.
    name: string, optional, default: 'Exp_EFM_Mod'

    References
    ----------
    [1] Yongfeng Zhang, Guokun Lai, Min Zhang, Yi Zhang, Yiqun Liu, and Shaoping Ma. 2014.
    Explicit factor models for explainable recommendation based on phrase-level sentiment analysis.
    https://doi.org/10.1145/2600428.2609579

    [2] https://github.com/PreferredAI/tutorials/blob/master/recommender-systems/07_explanations.ipynb

    """

    def __init__(self, rec_model, dataset, name="Exp_EFM_Mod"):
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
        threshold: float, optional, default:3.0
            Threshold for the aspect score to be considered in the explanation.

        Returns
        -------
        explanation: dict
            Explanations as a dictionary of aspect and score.
        """
        self.num_features = kwargs.get("feature_k", 3)
        self.threshold = kwargs.get("threshold", 3.0)
        user_id = self.dataset.uid_map[user_id]
        item_id = self.dataset.iid_map[item_id]

        id_aspect_map = {v: k for k, v in self.dataset.sentiment.aspect_id_map.items()}
        predicted_user_aspect_scores = np.dot(self.model.U1[user_id], self.model.V.T)
        predicted_item_aspect_scores = np.dot(self.model.U2[item_id], self.model.V.T)

        predicted_user_aspect_scores = self.normalize_rows(predicted_user_aspect_scores)
        predicted_item_aspect_scores = self.normalize_rows(predicted_item_aspect_scores)

        user_top_cared_aspects_ids = (-predicted_user_aspect_scores).argsort()[
            : self.num_features
        ]
        user_top_cared_aspects = [
            id_aspect_map[aid] for aid in user_top_cared_aspects_ids
        ]
        user_top_cared_aspects_score = predicted_item_aspect_scores[
            user_top_cared_aspects_ids
        ]

        # best_aspect = user_top_cared_aspects[predicted_item_aspect_scores[user_top_cared_aspects_ids].argmax()]
        # best_aspect_score = predicted_item_aspect_scores[user_top_cared_aspects_ids].max().astype(float)

        explanation = {}
        for i, aspect in enumerate(user_top_cared_aspects):
            if user_top_cared_aspects_score[i] >= self.threshold:
                explanation[aspect] = user_top_cared_aspects_score[i]
        return explanation

    def normalize_columns(self, matrix):
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

    def normalize_rows(self, array):
        # Compute the row-wise mean and standard deviation
        max_val = 5.0
        min_val = 0.0
        row_min = np.min(array)
        row_max = np.max(array)

        # Normalize the values of each row based on the minimum and maximum
        normalized_array = (array - row_min) / (row_max - row_min)
        scaled_array = normalized_array * (max_val - min_val) + min_val
        return scaled_array
