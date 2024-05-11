from .explainer import Explainer
import pandas as pd
import numpy as np
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import association_rules, apriori


class Exp_PHI4MF(Explainer):
    """Post Hoc Interpretability of Latent Factor Models for Recommendation Systems.

    Parameters
    ----------
    rec_model: object, recommender model
        The recommender model to be explained.
    dataset: object, dataset
        The dataset object that is used to explain.
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

    [2] https://github.com/ludovikcoba/recoxplainer/blob/master/recoxplainer/explain/post_hoc_association_rules.py
    """

    def __init__(
        self,
        rec_model,
        dataset,
        min_supp=0.001,
        min_conf=0.001,
        min_lift=0.01,
        name="Exp_PHI4MF",
    ):
        super().__init__(name=name, rec_model=rec_model, dataset=dataset)
        self.min_supp = min_supp
        self.min_conf = min_conf
        self.min_lift = min_lift

        if self.model is None:
            raise NotImplementedError("The model is None.")

        self.rules = self.generate_rules()

    def generate_rules(self):
        transactions = [
            [item for item in user_d[0]] for user_d in self.dataset.user_data.values()
        ]

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
        rules = rules[(rules["confidence"] > self.min_conf)]

        rules.consequents = [list(row.consequents)[0] for _, row in rules.iterrows()]
        rules.antecedents = [list(row.antecedents)[0] for _, row in rules.iterrows()]

        self.rules = rules[["consequents", "antecedents", "confidence"]]
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
            Explanations as a list of association rules
        """
        num_features = kwargs.get("feature_k", 10)
        uir_df = pd.DataFrame(
            np.array(self.dataset.uir_tuple).T, columns=["user", "item", "rating"]
        )
        uir_df["user"] = uir_df["user"].astype(int)
        uir_df["item"] = uir_df["item"].astype(int)
        if user_id not in self.dataset.uid_map:
            print(f"User {user_id} not in dataset")
            return []
        user_idx = self.dataset.uid_map[user_id]
        if item_id not in self.dataset.iid_map:
            print(f"Item {item_id} not in dataset")
            return []
        item_idx = self.dataset.iid_map[item_id]
        item_idx_2_id = {v: k for k, v in self.dataset.iid_map.items()}

        rules = self.rules[self.rules.consequents == item_idx]
        explanations = rules[
            rules.antecedents.isin(uir_df[uir_df.user == user_idx].item)
        ]
        explanations = explanations.sort_values(by=["confidence"], ascending=False)

        return [
            str(item_idx_2_id[x]) + "=>" + str(item_id)
            for x in explanations.antecedents[:num_features]
        ]
