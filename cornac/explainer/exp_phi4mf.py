from .explainer import Explainer
import pandas as pd
import numpy as np
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import association_rules, apriori


class Exp_PHI4MF(Explainer):
    """Post Hoc Interpretability of Latent Factor Models for Recommendation Systems.
    Explain by generating association rules from the recommendations of the model.

    Parameters
    ----------
    rec_model: object, recommender model
        The recommender model to be explained.
    dataset: object, dataset
        The dataset object that is used to explain.
    rec_k: int, optional, default: 10
        Number of recommendations to generate association rules.
    min_supp: float, optional, default: 0.001
        minimum support for the apriori algorithm
    min_conf: float, optional, default: 0.001
        minimum confidence for the apriori algorithm
    min_lift: float, optional, default: 0.01
        minimum lift for the apriori algorithm
    name: string, optional, default: 'Exp_PHI4MF'

    References
    ----------
    [1] Georgina Peake and Jun Wang. 2018. Explanation Mining: Post Hoc Interpretability of Latent Factor Models for Recommendation Systems.
    In Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, ACM, London United Kingdom, 2060-2069.
    DOI:https://doi.org/10.1145/3219819.3220072
    """

    def __init__(
        self,
        rec_model,
        dataset,
        rec_k=10,
        min_supp=0.001,
        min_conf=0.001,
        min_lift=0.01,
        name="Exp_PHI4MF",
    ):
        super().__init__(name=name, rec_model=rec_model, dataset=dataset)
        self.min_supp = min_supp
        self.min_conf = min_conf
        self.min_lift = min_lift
        self.rec_k = rec_k
        if self.model is None:
            raise NotImplementedError("The model is None.")

        self.rules = None
        self.item_idx_2_id = None
        self.uir_df = None 
        
    def generate_rules(self):
        transactions = []
        all_users = self.dataset.user_ids
        for user in all_users:
            item_ids = self.model.recommend(user, self.rec_k, remove_seen=False)
            # item_idx = [self.dataset.iid_map[item] for item in item_ids]
            transactions.append(item_ids)

        te = TransactionEncoder()
        te_transactions = te.fit_transform(transactions)
        te_transactions = pd.DataFrame(te_transactions, columns=te.columns_)
        itemsets = apriori(
            te_transactions,
            min_support=self.min_supp,
            use_colnames=True,
            low_memory=False,
        )
        rules = association_rules(itemsets, metric="lift", min_threshold=self.min_lift)
        rules = rules[rules["confidence"] > self.min_conf]
        rules = rules[rules["consequents"].apply(lambda x: len(list(x)) == 1)]
        rules["consequents"] = rules["consequents"].apply(lambda x: list(x)[0])
        
        self.rules = rules[["consequents", "antecedents", "confidence"]]
        print("Association rules generated")
        return rules

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
        explanations: list
            Explanations as a list of tuple (association rules, confidence).
        """
        if self.rules is None:
            self.generate_rules()
        
        feature_k = kwargs.get("feature_k", 10)
        
        if self.item_idx_2_id is None:
            self.item_idx_2_id = {v: k for k, v in self.dataset.iid_map.items()}
        if self.uir_df is None:
            self.uir_df = pd.DataFrame(
                        np.array(self.dataset.uir_tuple).T, columns=["user", "item", "rating"]
                    )
            self.uir_df["user"] = self.uir_df["user"].astype(int)
            self.uir_df["item"] = self.uir_df["item"].astype(int)
            self.uir_df["item_id"] = self.uir_df["item"].apply(lambda x: self.item_idx_2_id[x])
        if user_id not in self.dataset.uid_map:
            print(f"User {user_id} not in dataset")
            return []
        user_idx = self.dataset.uid_map[user_id]
        if item_id not in self.dataset.iid_map:
            print(f"Item {item_id} not in dataset")
            return []
        # item_idx = self.dataset.iid_map[item_id]
        
        user_items = set(self.uir_df[self.uir_df.user == user_idx]["item_id"])
        rules = self.rules[self.rules['consequents'] == item_id]
        rules = rules[rules['antecedents'].apply(lambda x: set(x).issubset(user_items))]
        if rules.empty:
            return []
        explanations = rules.sort_values(by=["confidence"], ascending=False)[:feature_k]
        return [
            (str(list(e['antecedents'])) + '=>' + str(item_id), e['confidence'])
            for _, e in explanations.iterrows()
        ]