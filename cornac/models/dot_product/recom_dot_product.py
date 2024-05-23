import numpy as np

from cornac.models import Recommender


class DotProduct(Recommender):
    def __init__(self, user_to_matrix_row, item_features, item_names, name="Dot Product Recommender"):
        super().__init__(name)
        self.user_to_matrix_row = user_to_matrix_row
        self.item_features = item_features
        self.item_names = item_names

    def score(self, user_feature, item_feature):
        """
        Parameters
        ----------
        user_feature : array-like
            The array containing the features of the user.

        item_feature : array-like
            The array containing the features of the item.
        """
        return np.dot(user_feature, item_feature)

    def recommend(self, user, top_k=1):
        """
        Parameters
        ----------
        user : str
            The ID of the user for whom recommendations are to be generated.
        top_k : int, optional
            The number of top recommendations to be returned. Default is 1.

        Returns
        -------
        recommendations : list of tuple
            A list of tuples containing the recommended item names and their corresponding scores.

        """
        user_feature = self.user_to_matrix_row[user]
        item_features = self.item_features
        # Compute scores for all items in one go using matrix multiplication
        scores = np.dot(user_feature, item_features.T)

        # Get the top_k item indices with the highest scores
        top_indices = np.argsort(-scores)[:top_k]

        # Prepare recommendations
        recommendations = [(self.item_names[index], scores[index]) for index in top_indices]
        return recommendations
