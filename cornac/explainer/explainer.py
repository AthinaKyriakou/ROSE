from tqdm.auto import tqdm
import pandas as pd

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
        explanation: dict
            Explanation for this recommendation. 
            
        Note
        ----
        If Explanation is not exp-score pair in dict, the `experiment/experiment_explainers.py` should be update to handle the format.
        """
        raise NotImplementedError("This method should be implemented by the subclass")


    def explain_one_with_ref(self, user_id, item_id, ref_item_id, **kwargs):
        """Provide explanation for one user, one item and another explanation for reference item. 
        
        Parameters
        ----------
        user_id: str
            One user's id.
        item_id: str
            One item's id.
        ref_item_id: str
            One reference item's id.
            
        Returns
        -------
        explanation: dataframe
            Explanation in columns [user_id, item_id, explanation, ref_item_id, ref_explanation]
        """
        
        exp = self.explain_one_recommendation_to_user(user_id, item_id, **kwargs)
        ref_exp = self.explain_one_recommendation_to_user(user_id, ref_item_id, **kwargs)
        
        explanation = {
            "user_id": user_id,
            "item_id": item_id,
            "explanation": exp,
            "ref_item_id": ref_item_id,
            "ref_explanation": ref_exp
        }
        explanation = pd.DataFrame(explanation)

        return explanation