from typing import Dict

import numpy as np
from scipy.optimize import minimize

from cornac.explainer import Explainer
from cornac.models import Recommender


class CounterExplainer(Explainer):
    """
    Counterfactual Explainable Recommendation (CountER)

    Parameters
    ----------
    model (Recommender): The recommender model.
    user_to_matrix_row (Dict[str, np.ndarray]): Mapping from user to their feature vector.
    item_aspect_matrix (Dict[str, np.ndarray]): Mapping from item to its feature vector.
    alpha (float): The alpha parameter for the objective function.
    gamma (float): The gamma parameter for the objective function.
    lambda_param (float): The lambda parameter for the objective function.

    References
    ----------
    Juntao Tan, Shuyuan Xu, Yingqiang Ge, Yunqi Li, Xu Chen, Yongfeng Zhang.
    2021. Counterfactual Explainable Recommendation. In Proceedings of the
    30th ACM International Conference on Information and Knowledge Management (CIKM ’21),
    November 1–5, 2021, Virtual Event, QLD, Australia. ACM,
    New York, NY, USA, 10 pages. https://doi.org/10.1145/3459637.3482420

    Code Reference
    ----------
    https://github.com/chrisjtan/counter

    """

    def __init__(self, model, user_to_matrix_row: Dict[str, np.ndarray],
                 item_to_matrix_row: Dict[str, np.ndarray], alpha: float = 0.2,
                 gamma: float = 1.0, lambda_param: float = 100):
        """
        Initialize the CounterExplainer.

        Args:
            model (Recommender): The recommender model.
            user_to_matrix_row (Dict[str, np.ndarray]): Mapping from user to their feature vector.
            item_to_matrix_row (Dict[str, np.ndarray]): Mapping from item to its feature vector.
            alpha (float): The alpha parameter for the objective function.
            gamma (float): The gamma parameter for the objective function.
            lambda_param (float): The lambda parameter for the objective function.
        """
        self._validate_init_preconditions(alpha, gamma, item_to_matrix_row, lambda_param, model, user_to_matrix_row)
        self.model = model
        self.user_to_matrix_row = user_to_matrix_row
        self.item_aspect_matrix = item_to_matrix_row
        self.alpha = alpha
        self.gamma = gamma
        self.lambda_param = lambda_param

    def _validate_init_preconditions(self, alpha, gamma, item_to_matrix_row, lambda_param, model, user_to_matrix_row):
        assert model is not None, "Model is None"
        assert user_to_matrix_row is not None, "User to matrix row is None"
        assert item_to_matrix_row is not None, "Item to matrix row is None"
        assert alpha is not None, "Alpha is None"
        assert gamma is not None, "Gamma is None"
        assert lambda_param is not None, "Lambda is None"
        assert alpha >= 0, "Alpha is negative"
        assert gamma >= 0, "Gamma is negative"
        assert lambda_param >= 0, "Lambda is negative"
        assert 0 <= alpha <= 1, "Alpha is not in [0, 1]"
        assert model.score is not None, "Model has no score function"
        assert user_to_matrix_row.items() != 0, "User to matrix row is empty"
        assert item_to_matrix_row.items() != 0, "Item to matrix row is empty"

    def _objective_function(self, delta: np.ndarray, user_feature_vector: np.ndarray, item_feature_vector: np.ndarray,
                            item_kplus1_feature_vector: np.ndarray) -> float:
        """Objective function to be minimized. See equation 9 in paper"""
        # Variable names were chosen to match the paper
        s_ij_delta = self.model.score(user_feature_vector, item_feature_vector + delta)
        s_ij_kplus1 = self.model.score(user_feature_vector, item_kplus1_feature_vector)
        l1_norm = np.linalg.norm(delta, 1)
        l2_squared = np.linalg.norm(delta, 2) ** 2
        hinge_loss = np.maximum(0, self.alpha + s_ij_delta - s_ij_kplus1)
        return l2_squared + self.gamma * l1_norm + self.lambda_param * hinge_loss

    def explain_one_recommendation_to_user(self, user: str, item: str, second_best_item: str) -> np.ndarray:
        """
        Explain a recommendation to a user.

        Args:
            user (str): The user.
            item (str): The item.
            second_best_item (str): The second best item.

        Returns:
            np.ndarray: The delta vector that explains the recommendation.
        """
        self._validate_explain_preconditions(item, second_best_item, user)

        user_feature_vector = self.user_to_matrix_row[user]
        item_feature_vector = self.item_aspect_matrix[item]
        second_best_item_feature_vector = self.item_aspect_matrix[second_best_item]

        initial_delta = np.zeros(user_feature_vector.shape)
        bounds = self._initialize_bounds(item_feature_vector, user_feature_vector)
        if bounds == [] or bounds == [(0, 0)] * len(bounds):
            return initial_delta

        result = self._optimize_objective_function(bounds, initial_delta, item_feature_vector,
                                                   second_best_item_feature_vector, user_feature_vector)

        return self._apply_threshold(result.x)

    def _validate_explain_preconditions(self, item, second_best_item, user):
        assert user is not None, "User is None"
        assert item is not None, "Item is None"
        assert second_best_item is not None, "Second best item is None"
        assert user in self.user_to_matrix_row, f"User {user} not found in the user_to_matrix_row"
        assert item in self.item_aspect_matrix, f"Item {item} not found in the item_aspect_matrix"
        assert self.user_to_matrix_row[user].shape == self.item_aspect_matrix[item].shape, "User and item have different feature vector sizes"
        assert second_best_item in self.item_aspect_matrix, f"Item {second_best_item} not found in item_aspect_matrix"
        assert item != user, "Item is the user"
        assert item != second_best_item, "Items should be different"

    def _initialize_bounds(self, item_feature_vector, user_feature_vector):
        """
        Creates bounds for the optimization based on shared feature vectors.
        """
        return [
            (-item_feature_vector[i], 0) if user_feature_vector[i] != 0 and item_feature_vector[i] != 0 else (0, 0)
            for i in range(len(item_feature_vector))]

    def _optimize_objective_function(self, bounds, initial_delta, item_feature_vector, second_best_item_feature_vector,
                                     user_feature_vector, optimization_method='SLSQP'):
        """
        Minimizes the objective function to find the explanation delta.
        """
        return minimize(self._objective_function, initial_delta,
                          args=(user_feature_vector, item_feature_vector, second_best_item_feature_vector),
                          method=optimization_method, bounds=bounds, options={'disp': False})

    def _apply_threshold(self, delta, threshold=0.001):
        """
        Applies a threshold to delta to filter out negligible changes.
        """
        delta[np.abs(delta) < threshold] = 0
        return delta
