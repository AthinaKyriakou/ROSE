from tqdm.auto import tqdm

class Explainer:
    def __init__(self, name, rec_model, dataset):
        """
        rec_model: trained recommendation model
        dataset: dataset used for explanation
        """
        self.name = name
        self.model = rec_model 
        self.recommendations = None
        self.dataset = dataset
        self.num_items = dataset.num_items
        self.num_users = dataset.num_users

    def explain_recommendations(self, recommendations, num_features=10):
        """"
        recommendations: dataframe, columns name as [user_id, item_id]
        """
        explanations = []
        self.recommendations = recommendations 

        with tqdm(total=self.recommendations.shape[0], desc="Computing explanations: ") as pbar:

            for _, row in self.recommendations.iterrows():
                explanations.append(self.explain_one_recommendation_to_user(row.user_id, row.item_id, num_features))
                pbar.update()

        self.recommendations['explanations'] = explanations
        return self.recommendations

    def explain_one_recommendation_to_user(self, user_id, item_id, num_features=10):
        """"
        provide explanation for one user and one item
        user_id: one user
        item_id: one item
        """
        pass

