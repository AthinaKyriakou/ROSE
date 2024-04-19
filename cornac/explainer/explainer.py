from tqdm.auto import tqdm


class Explainer:
    def __init__(self, name, rec_model, dataset):
        """Generic class for a Explainer. All explainers should inherit from this class.

        Parameters
        ----------
        name: string
            Name of the explainer.
        rec_model: obj: `cornac.models.Recommender`, required
            Recommender model to be explained.
        dataset: obj: `cornac.data.Dataset`, required
            Dataset object that is used to explain.
        """
        self.name = name
        self.model = rec_model
        self.recommendations = None
        self.dataset = dataset
        self.num_items = dataset.num_items
        self.num_users = dataset.num_users

    def explain_recommendations(self, recommendations, **kwargs):
        """Explains a list of recommendations. Using the explain_one_recommendation_to_user method to explain each recommendation.

        Parameters
        ----------
        recommendations: pandas.DataFrame columns name as [user_id, item_id]
            List of recommendations to be explained.

        Returns
        -------
        recommendations: pandas.DataFrame
            List of recommendations with explanations.
        """
        explanations = []
        self.recommendations = recommendations

        with tqdm(
            total=self.recommendations.shape[0], desc="Computing explanations: "
        ) as pbar:

            for _, row in self.recommendations.iterrows():
                explanations.append(
                    self.explain_one_recommendation_to_user(
                        row.user_id, row.item_id, **kwargs
                    )
                )
                pbar.update()

        self.recommendations["explanations"] = explanations
        return self.recommendations

    def explain_one_recommendation_to_user(self, user_id, item_id, **kwargs):
        """Provide explanation for one user and one item. Each explainer should implement this method.

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
        Explanation for this recommendation.
        """
        raise NotImplementedError("This method should be implemented by the subclass")
