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

class LimeRSExplainer(Explainer):
    """Local Interpretable Model-agnostic Explainer for Recommender System
    
    Reference:
    -------------
    Nóbrega, C., &amp; Marinho, L. (2019). Towards explaining recommendations through local surrogate models. 
    Proceedings of the 34th ACM/SIGAPP Symposium on Applied Computing. doi:10.1145/3297280.3297443
    
    """
    def __init__(
        self,
        rec_model,
        dataset,
        num_samples = 100,
        name="LIMERS",
        mode="regression",
        kernel_width=25,
        verbose=False,
        class_names=np.array(["rec"]),
        feature_selection="highest_weights",
        random_state=None,
    ):
        """
        Args:
            rec_model: recommendation model
            dataset: dataset used for training
            mode: "regression", "classification". Note classification is not implemented
            kernel_width: used to fit kernel function
            verbose: (LIME base) if true, print local prediction values from linear model.
            class_names: (LIME base) list of class names (only used for classification)
            feature_selection: (LIME base) how to select num_features. options are:
                'forward_selection': iteratively add features to the model.
                    This is costly when num_features is high
                'highest_weights': selects the features that have the highest
                    product of absolute weight * original data point when
                    learning with all the features
                'lasso_path': chooses features based on the lasso
                    regularization path
                'none': uses all features, ignores num_features
                'auto': uses forward_selection if num_features <= 6, and
                    'highest_weights' otherwise.
            random_state: (LIME base) an integer or numpy.RandomState that will be used to
                generate random numbers. If None, the random state will be
                initialized using the internal numpy seed.

        """
        super(LimeRSExplainer, self).__init__(name, rec_model, dataset)
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
            self, 
            recommendations, 
            num_features=10,
            feature_type = "features",
            verbose = True
        ):
        """
        Generate explanations for a list of recommendations
        Args:
            recommendations: df of [user_id, item_id]
            num_samples: number of datapoint to be sampled
            num_features: number of features used in the explanation
            feature_type: default "features"
        """
        if self.feature_map == None:
            self.feature_map = {i: self.model.one_hot_columns[i] for i in range(len(list(self.model.one_hot_columns)))}
        if self.n_rows == None:
            self.n_rows = self.model.train_set.num_ratings
            self.user_freq = self.model.train_set.training_df["user_id"].value_counts(normalize=True)  
            self.item_freq = self.model.train_set.training_df["item_id"].value_counts(normalize=True)

        explanations = pd.DataFrame(columns=['user_id', 'item_id', 'explanations', 'local_prediction'])
        self.recommendations = recommendations 

        if verbose:
            total = self.recommendations.shape[0]
            interval = int(total * 0.1)
            with tqdm(total=total, desc="Computing explanations: ") as pbar:

                for i, row in self.recommendations.iterrows():
                    explanations = pd.concat(
                        [explanations, 
                        self.explain_one_recommendation_to_user(row.user_id, row.item_id, 
                                                                                num_features, feature_type, verbose)])
                    if i % interval == 0:
                        pbar.update(interval)
        else:
            for _, row in self.recommendations.iterrows():
                explanations = pd.concat(
                    [explanations, 
                    self.explain_one_recommendation_to_user(row.user_id, row.item_id, 
                                                                            num_features, feature_type, verbose)])

        return explanations
    
    def explain_one_recommendation_to_user(
            self,
            user_id,
            item_id,
            num_features = 10,
            feature_type="features",
            verbose = True
    ):
        """
        Generate a list of explanations for each instance

        Args:
            user_id: id of user to be explained
            item_id: id of item to be explained
            num_features: num of features to be included in the output
            feature_type: "features" as default

        Return: Dataframe of explanations [user_id, item_id, explanations, local_prediction] for each instance

        """
        if not self.model.train_set.item_features.empty and self.model.train_set.user_features.empty:
            user_idx = self.dataset.uid_map[user_id]
            item_idx = self.model.train_set.iid_map[item_id]
        elif not self.model.train_set.user_features.empty and self.model.train_set.item_features.empty:
            user_idx = self.model.train_set.uid_map[user_id]
            item_idx = self.dataset.iid_map[item_id]
        elif not self.model.train_set.user_features.empty and not self.model.train_set.item_features.empty:
            try:
                user_idx = self.model.train_set.uid_map[user_id]
                item_idx = self.model.train_set.iid_map[item_id]
            except KeyError:
                output_df = pd.DataFrame({
                    "user_id": [user_id],
                    "item_id": [item_id],
                    "explanations": [],
                    "local_prediction": []})
                return output_df
        else:
            raise ValueError("LIMERS requires at least one of item features or user features!")

        if self.feature_map == None:
            self.feature_map = {i: self.model.one_hot_columns[i] for i in range(len(list(self.model.one_hot_columns)))}
        
        if self.n_rows == None:
            self.n_rows = self.model.train_set.num_ratings
            self.user_freq = self.model.train_set.training_df["user_id"].value_counts(normalize=True)  
            self.item_freq = self.model.train_set.training_df["item_id"].value_counts(normalize=True)
        
        # if verbose:
        #     logger.info("explaining-> (user: {}, item: {})".format(user_id, item_id))
            
        exp = self.explain_instance(
                user_idx, item_idx, num_features=num_features, 
                neighborhood_entity="item", labels=[0], )
        
        
        filtered_features = self.extract_features(
                exp.local_exp[0],
                feature_type=feature_type,
            )

        #explanation_str = json.dumps(filtered_features)
        output_df = pd.DataFrame({
                "user_id": [user_id],
                "item_id": [item_id],
                "explanations": [filtered_features],
                "local_prediction": [round(exp.local_pred[0],3)],
        })
            # result.append(output_df)

        return output_df
    
    def extract_features(
            self,
            explanation_all_ids,
            feature_type="features"
    ):
        """
        Args:
            explanation_all_ids: feature ids (x) of sorted list of tuples (x,y). Sorted in decreasing absolute value of local weight (y)
            feature_type: if "features", extract features names as explanation only; if "item", extract item_id as explanation only

        Return:
        """
        filtered_dict = dict()
        if feature_type == "features":
            for tup in explanation_all_ids:
                if not (
                    self.feature_map[tup[0]].startswith("user_id")
                    or self.feature_map[tup[0]].startswith("item_id")
                ):
                    filtered_dict[self.feature_map[tup[0]]] = round(tup[1],6)
        elif feature_type == "item":
            top_features = 50
            for tup in explanation_all_ids:
                if(
                    self.feature_map[tup[0]].startwith("item_id")
                    and len(filtered_dict) <= top_features
                ):
                    filtered_dict[self.feature_map[tup[0]]] = round(tup[1],6)
        return filtered_dict

    def explain_instance(
        self,
        user_idx,
        item_idx,
        neighborhood_entity,
        num_features,
        labels=(1,),
        distance_metric="cosine",
        model_regressor=None,
    ):
        """
        Provide explanations for one instance of (user_id, item_id)

        Args:
            instance: an instance (user_id, item_id) to be explained 
            neighborhood_entity: rule for selecting samples. Option: "user", "item", or ??
            labels: only the first is used for regression task
            num_features: number of explained features to be displayed
            num_samples: number of samples selected in the neighbourhood
            distance_metric: default "cosine", used to calculate pairwise similarity
            model_regressor: (LIME base) sklearn regressor to use in explanation.
                Defaults to Ridge regression if None. Must have
                model_regressor.coef_ and 'sample_weight' as a parameter
                to model_regressor.fit()

        Returns:
            LIME Explanation class

        """
        # get neighborhood
        neighborhood_df = self.generate_neighborhood(
            user_idx, item_idx, neighborhood_entity, self.num_samples
        )  
        
        data = self.model.train_set.merge_uir_with_features(neighborhood_df, self.model.uses_features)
        # compute distance based on interpretable format
        data, _ = self.model.train_set.convert_to_pyfm_format(
                data, columns=self.model.one_hot_columns
        )
        distances = sklearn.metrics.pairwise_distances(
                data, data[0].reshape(1, -1), metric=distance_metric
        ).ravel() 

        if self.model.name == 'fm_regressor' or self.model.name == 'ffm_regressor': 
            # get predictions from original complex model
            yss = self.model.score_neighbourhood(neighborhood_df)

        else:
            raise NotImplementedError(
                "LIME-RS only support FM and FFM models."
            )

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
            (ret_exp.intercept[label],  ret_exp.local_exp[label], 
             ret_exp.score,ret_exp.local_pred,) = self.base.explain_instance_with_data(  
                data,  yss,  distances,  label,  num_features,  
                model_regressor=model_regressor, 
                feature_selection=self.feature_selection,  
            )

        return ret_exp

    def generate_neighborhood(self, user_idx, item_idx, entity, num_samples):  
        """
        Args:
            instance: instance to be explained
            entity: rule for selecting samples. Option: "user", "item"
            num_samples: number of samples selected in the neighborhood

        Return:
            Dataframe of user_id, item_id
        """
        samples = list()
        samples.append(
            {"user_id": str(user_idx), "item_id": str(item_idx)}
        )
        if entity == "user":
            sample_users = np.random.choice(
                self.user_freq.index.tolist(),
                num_samples - 1,
                replace=False,
                p=self.user_freq.values.tolist(),
            )
            for u in sample_users:
                samples.append({"user_id": str(u), "item_id": str(item_idx)})

        elif entity == "item":
            sample_items = np.random.choice(
                self.item_freq.index.tolist(),
                num_samples - 1,
                replace=False,
                p=self.item_freq.values.tolist(),
            )
            for i in sample_items:  
                samples.append({"user_id": str(user_idx), "item_id": str(i)})
        else:
            sample_rows = np.random.choice(
                range(self.n_rows), num_samples - 1, replace=False
            )
            for s in self.model.train_set.training_df.iloc[sample_rows].itertuples():
                samples.append({"user_id": str(s.user_id), "item_id": str(s.item_id)})

        samples_df = pd.DataFrame(samples)
        samples_df = samples_df[["user_id", "item_id"]]

        return samples_df 