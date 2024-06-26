import json
import logging
from tqdm.auto import tqdm

import numpy as np
import pandas as pd
import sklearn
from lime import lime_base, explanation
from sklearn.utils import check_random_state

from cornac.explainer import Explainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Exp_LIMERS(Explainer):
    """Local Interpretable Model-agnostic Explainer for Recommender System

    Parameters
    ----------
    rec_model: object, recommender model
        The recommender model to be explained.
    dataset: object, dataset
        The dataset object that is used to explain.
    num_samples: int, optional, default: 100
        Number of samples to be generated for explanation
    name: string, optional, default: 'Exp_LIMERS'
    mode: string, optional, default: 'regression'
        The mode of the explainer. Options: 'regression', 'classification'
        Note: 'classification' is not implemented
    kernel_width: int, optional, default: 25
        Used to fit kernel function
    verbose: bool, optional, default: False
        (LIME base) If true, print local prediction values from linear model.
    class_names: list, optional, default: np.array(["rec"])
        (LIME base) List of class names (only used for classification)
    feature_selection: string, optional, default: 'highest_weights'
        (LIME base) How to select feature_k. Options are:
            - 'forward_selection': iteratively add features to the model. This is costly when feature_k is high
            - 'highest_weights': selects the features that have the highest product of absolute weight * original data point when learning with all the features
            - 'lasso_path': chooses features based on the lasso regularization path
            - 'none': uses all features, ignores feature_k
            - 'auto': uses forward_selection if feature_k <= 6, and 'highest_weights' otherwise.
    random_state: int or numpy.RandomState, optional, default: None
        (LIME base) An integer or numpy.RandomState that will be used to
        generate random numbers. If None, the random state will be
        initialized using the internal numpy seed.

    References
    ----------
    [1] Caio Nóbrega and Leandro Marinho. 2019. Towards explaining recommendations through local surrogate models. https://doi.org/10.1145/3297280.3297443

    """

    def __init__(
        self,
        rec_model,
        dataset,
        num_samples=100,
        name="Exp_LIMERS",
        mode="regression",
        kernel_width=25,
        verbose=False,
        class_names=np.array(["rec"]),
        feature_selection="highest_weights",
        random_state=None,
    ):
        super(Exp_LIMERS, self).__init__(name, rec_model, dataset)

        # exponential kernel
        def kernel(d):
            return np.sqrt(np.exp(-(d**2) / kernel_width**2))

        self.random_state = check_random_state(random_state)
        self.base = lime_base.LimeBase(kernel, verbose, random_state=self.random_state)
        self.feature_map = None
        self.mode = mode
        self.class_names = class_names
        self.feature_selection = feature_selection
        self.dataset = dataset
        self.model = rec_model
        self.num_samples = num_samples
        self.n_rows = None
        self.user_freq = None
        self.item_freq = None

    @staticmethod
    def convert_and_round(values):
        return ["%.2f" % v for v in values]

    def explain_recommendations(
        self, recommendations, feature_k=10, feature_type="features", verbose=True
    ):
        """Generate explanations for a list of recommendations

        Parameters
        ----------
            recommendations: pandas.DataFrame, required
                Dataframe of [user_id, item_id]
            feature_k: int, optional, default: 10
                Number of features to be included in the output
            feature_type: string, optional, default: "features"
                If "features", extract features names as explanation only; if "item", extract item_id as explanation only
            verbose: bool, optional, default: True

        Returns
        -------
            explanations: pandas.DataFrame
                Dataframe of explanations [user_id, item_id, explanations, local_prediction] for each instance
        """
        if self.feature_map == None:
            self.feature_map = {
                i: self.model.one_hot_columns[i]
                for i in range(len(list(self.model.one_hot_columns)))
            }
        if self.n_rows == None:
            self.n_rows = self.model.train_set.num_ratings
            self.user_freq = self.model.train_set.training_df["user_id"].value_counts(
                normalize=True
            )
            self.item_freq = self.model.train_set.training_df["item_id"].value_counts(
                normalize=True
            )

        explanations = pd.DataFrame(
            columns=["user_id", "item_id", "explanations", "local_prediction"]
        )
        self.recommendations = recommendations

        if verbose:
            total = self.recommendations.shape[0]
            interval = int(total * 0.1)
            with tqdm(total=total, desc="Computing explanations: ") as pbar:

                for i, row in self.recommendations.iterrows():
                    explanations = pd.concat(
                        [
                            explanations,
                            self.explain_one_recommendation_to_user(
                                row.user_id,
                                row.item_id,
                                feature_k,
                                feature_type,
                                verbose,
                            ),
                        ]
                    )
                    if i % interval == 0:
                        pbar.update(interval)
        else:
            for _, row in self.recommendations.iterrows():
                explanations = pd.concat(
                    [
                        explanations,
                        self.explain_one_recommendation_to_user(
                            row.user_id,
                            row.item_id,
                            feature_k,
                            feature_type,
                            verbose,
                        ),
                    ]
                )

        return explanations

    def explain_one_recommendation_to_user(
        self, user_id, item_id, feature_k=10, feature_type="features", verbose=True
    ):
        """Generate explanations for one recommendation

        Parameters
        ----------
            user_id: int, required
                id of user to be explained
            item_id: int, required
                id of item to be explained
            feature_k: int, optional, default: 10
                Number of features to be included in the output
            feature_type: string, optional, default: "features"
                If "features", extract features names as explanation only; if "item", extract item_id as explanation only
            verbose: bool, optional, default: True

        Returns
        -------
            explanation: pandas.DataFrame
                Dataframe of explanations [user_id, item_id, explanations, local_prediction] for one instance
        """
        if (
            not self.model.train_set.item_features.empty
            and self.model.train_set.user_features.empty
        ):
            user_idx = self.dataset.uid_map[user_id]
            item_idx = self.model.train_set.iid_map[item_id]
        elif (
            not self.model.train_set.user_features.empty
            and self.model.train_set.item_features.empty
        ):
            user_idx = self.model.train_set.uid_map[user_id]
            item_idx = self.dataset.iid_map[item_id]
        elif (
            not self.model.train_set.user_features.empty
            and not self.model.train_set.item_features.empty
        ):
            try:
                user_idx = self.model.train_set.uid_map[user_id]
                item_idx = self.model.train_set.iid_map[item_id]
            except KeyError:
                explanation = pd.DataFrame(
                    {
                        "user_id": [user_id],
                        "item_id": [item_id],
                        "explanations": [],
                        "local_prediction": [],
                    }
                )
                return explanation
        else:
            raise ValueError(
                "LIMERS requires at least one of item features or user features!"
            )

        if self.feature_map == None:
            self.feature_map = {
                i: self.model.one_hot_columns[i]
                for i in range(len(list(self.model.one_hot_columns)))
            }

        if self.n_rows == None:
            self.n_rows = self.model.train_set.num_ratings
            self.user_freq = self.model.train_set.training_df["user_id"].value_counts(
                normalize=True
            )
            self.item_freq = self.model.train_set.training_df["item_id"].value_counts(
                normalize=True
            )

        # if verbose:
        #     logger.info("explaining-> (user: {}, item: {})".format(user_id, item_id))

        exp = self.explain_instance(
            user_idx,
            item_idx,
            feature_k=feature_k,
            neighborhood_entity="item",
            labels=[0],
        )

        filtered_features = self.extract_features(
            exp.local_exp[0],
            feature_type=feature_type,
        )

        # explanation_str = json.dumps(filtered_features)
        explanation = pd.DataFrame(
            {
                "user_id": [user_id],
                "item_id": [item_id],
                "explanations": [filtered_features],
                "local_prediction": [round(exp.local_pred[0], 3)],
            }
        )
        # result.append(output_df)

        return explanation

    def extract_features(self, explanation_all_ids, feature_type="features"):
        """Extract features from explanation

        Parameters
        ----------
        explanation_all_ids: list, required
            Feature ids (x) of sorted list of tuples (x,y). Sorted in decreasing absolute value of local weight (y)
        feature_type: string, optional, default: "features"
            If "features", extract features names as explanation only; if "item", extract item_id as explanation only

        Returns
        -------
        filtered_dict: dict
            Dictionary of features and their weights
        """
        filtered_dict = dict()
        if feature_type == "features":
            for tup in explanation_all_ids:
                if not (
                    self.feature_map[tup[0]].startswith("user_id")
                    or self.feature_map[tup[0]].startswith("item_id")
                ):
                    filtered_dict[self.feature_map[tup[0]]] = round(tup[1], 6)
        elif feature_type == "item":
            top_features = 50
            for tup in explanation_all_ids:
                if (
                    self.feature_map[tup[0]].startwith("item_id")
                    and len(filtered_dict) <= top_features
                ):
                    filtered_dict[self.feature_map[tup[0]]] = round(tup[1], 6)
        return filtered_dict

    def explain_instance(
        self,
        user_idx,
        item_idx,
        neighborhood_entity,
        feature_k=10,
        labels=(1,),
        distance_metric="cosine",
        model_regressor=None,
    ):
        """Provide explanations for one instance of (user_idx, item_idx)

        Parameters
        ----------
        user_idx: int, required
            Index of user to be explained
        item_idx: int, required
            Index of item to be explained
        neighborhood_entity: string, optional, default: "item"
            Rule for selecting samples. Options: "user", "item", or ??
        labels: list, optional, default: (1,)
            Only the first is used for regression task
        feature_k: int, optional, default: 10
            Number of explained features to be displayed
        distance_metric: string, optional, default: "cosine"
            Used to calculate pairwise similarity
        model_regressor: sklearn regressor, optional, default: None
            (LIME base) sklearn regressor to use in explanation.
            Defaults to Ridge regression if None. Must have
            model_regressor.coef_ and 'sample_weight' as a parameter
            to model_regressor.fit()

        Returns
        -------
        ret_exp: object, lime.explanation
            Explanation object
        """
        # get neighborhood
        neighborhood_df = self.generate_neighborhood(
            user_idx, item_idx, neighborhood_entity, self.num_samples
        )

        data = self.model.train_set.merge_uir_with_features(
            neighborhood_df, self.model.uses_features
        )
        # compute distance based on interpretable format
        data, _ = self.model.train_set.convert_to_pyfm_format(
            data, columns=self.model.one_hot_columns
        )
        distances = sklearn.metrics.pairwise_distances(
            data, data[0].reshape(1, -1), metric=distance_metric
        ).ravel()

        if self.model.name == "FMRec" or self.model.name == "ffm_regressor":
            # get predictions from original complex model
            yss = self.model.score_neighbourhood(neighborhood_df)

        else:
            raise NotImplementedError("LIME-RS only support FM and FFM models.")

        # for classification, the model needs to provide a list of tuples - classes along with prediction probabilities
        if self.mode == "classification":
            raise NotImplementedError(
                "LIME-RS does not currently support classifier models."
            )
        # for regression, the output should be a one-dimensional array of predictions
        else:
            try:
                assert isinstance(yss, np.ndarray) and len(yss.shape) == 1
            except AssertionError:
                raise ValueError(
                    "Your model needs to output single-dimensional \
                            numpyarrays, not arrays of {} dimensions".format(
                        yss.shape
                    )
                )

            predicted_value = yss[0]
            min_y = min(yss)
            max_y = max(yss)

            # add a dimension to be compatible with downstream machinery
            yss = yss[:, np.newaxis]

        ret_exp = explanation.Explanation(
            domain_mapper=None, mode=self.mode, class_names=self.class_names
        )
        if self.mode == "classification":
            raise NotImplementedError(
                "LIME-RS does not currently support classifier models."
            )
        else:
            ret_exp.predicted_value = predicted_value
            ret_exp.min_value = min_y
            ret_exp.max_value = max_y
            labels = [0]

        for label in labels:
            (
                ret_exp.intercept[label],
                ret_exp.local_exp[label],
                ret_exp.score,
                ret_exp.local_pred,
            ) = self.base.explain_instance_with_data(
                data,
                yss,
                distances,
                label,
                feature_k,
                model_regressor=model_regressor,
                feature_selection=self.feature_selection,
            )

        return ret_exp

    def generate_neighborhood(self, user_idx, item_idx, entity, num_samples):
        """Generate neighborhood samples for explanation

        Parameters
        ----------
        user_idx: int, required
            Index of user to be explained
        item_idx: int, required
            Index of item to be explained
        entity: string, optional, default: "item"
            Rule for selecting samples. Options: "user", "item"
        num_samples: int, optional, default: 100
            Number of samples to be generated for explanation

        Returns
        -------
        samples_df: pandas.DataFrame
            Dataframe of user_id, item_id
        """
        samples = list()
        samples.append({"user_id": str(user_idx), "item_id": str(item_idx)})
        if entity == "user":
            if len(self.user_freq.index.tolist()) < num_samples:
                raise ValueError(
                    f"You want to choice {num_samples} user samples, but you only have {len(self.user_freq.index.tolist())} users."
                )
            sample_users = np.random.choice(
                self.user_freq.index.tolist(),
                num_samples - 1,
                replace=False,
                p=self.user_freq.values.tolist(),
            )
            for u in sample_users:
                samples.append({"user_id": str(u), "item_id": str(item_idx)})

        elif entity == "item":
            if len(self.item_freq.index.tolist()) < num_samples:
                raise ValueError(
                    f"You want to choice {num_samples} item samples, but you only have {len(self.item_freq.index.tolist())} items."
                )
            sample_items = np.random.choice(
                self.item_freq.index.tolist(),
                num_samples - 1,
                replace=False,
                p=self.item_freq.values.tolist(),
            )
            for i in sample_items:
                samples.append({"user_id": str(user_idx), "item_id": str(i)})
        else:
            if self.n_rows < num_samples:
                raise ValueError(
                    f"You want to choice {num_samples} samples, but you only have {self.n_rows} ratings."
                )
            sample_rows = np.random.choice(
                range(self.n_rows), num_samples - 1, replace=False
            )
            for s in self.model.train_set.training_df.iloc[sample_rows].itertuples():
                samples.append({"user_id": str(s.user_id), "item_id": str(s.item_id)})

        samples_df = pd.DataFrame(samples)
        samples_df = samples_df[["user_id", "item_id"]]

        return samples_df
