import numpy as np
cimport numpy as np
import pandas as pd
cimport cython
from tqdm.auto import tqdm

from ..metric_exp import Metric_Exp


class Metric_Exp_FPR(Metric_Exp):
    """probability of feature precision and recall
    
    Parameters
    ----------
    rec_k: int, optional, default: 10
        The number of items to recommend for each user.
    feature_k: int, optional, default: 10
        The number of features to explain for each user-item pair.
    ground_truth: pd.DataFrame, optional, default: None
        The ground truth explanations for each user-item pair, columns=['user_id', 'item_id', 'explanations']

    References
    ----------
    [1]Juntao Tan, Shuyuan Xu, Yingqiang Ge, Yunqi Li, Xu Chen, and Yongfeng Zhang. 2021. Counterfactual Explainable Recommendation. In Proceedings of the 30th ACM International Conference on Information & Knowledge Management (CIKM '21). https://doi.org/10.1145/3459637.3482420

    """

    def __init__(self, name="Metric_Exp_FPR", rec_k=10, feature_k=10, ground_truth=None):
        super().__init__(name=name, rec_k=rec_k, feature_k=feature_k)
        self.ground_truth = ground_truth

    def _create_ground_truth_sentiment(self):
        if self.ground_truth is None:
            raise ValueError("Please provide the ground truth.")
        u_i_all = pd.DataFrame(self.ground_truth, columns=['user_id', 'item_id', 'explanations'])
        u_i_all['explanations'] = u_i_all['explanations'].apply(self._transform_format)
        user_id_list = self.model.train_set.uid_map.keys()
        item_id_list = self.model.train_set.iid_map.keys()
        selected_indices = []
        for i, row in u_i_all.iterrows():
            if u_i_all.at[i, 'user_id'] in user_id_list and u_i_all.at[i, 'item_id'] in item_id_list:
                selected_indices.append(i)
        u_i_gt = u_i_all.iloc[selected_indices]
        return u_i_gt

    def _transform_format(self, lexicon):
        # Transform each tuple into the desired format
        cdef dict explanation_dict = {}
        for t in lexicon:
            if int(t[2])== 1:
                explanation_dict[t[0]] = int(t[2])
        # Join the tuples into a comma-separated string
        return explanation_dict

    def _creat_grount_truth_efm(self):
        A, X, Y = self.model._build_matrices(self.model.train_set)
        user_idx2id = {v: k for k, v in self.model.train_set.uid_map.items()}
        item_idx2id = {v: k for k, v in self.model.train_set.iid_map.items()}
        id2aspect = {v:k for k, v in self.model.train_set.sentiment.aspect_id_map.items()}
        users = []
        items = []
        explanations = []
        for i in range(A.shape[0]):
            for j in A[i,:].nonzero()[1]:
                u_aspect_indices = X[i, :].nonzero()[1]
                u_aspect_values = X[i, :].data
                i_aspect_ids = Y[j, :].nonzero()[1]
                i_aspect_values = Y[j, :].data

                aspect_ids = np.hstack((u_aspect_indices, i_aspect_ids))
                aspect_values = np.hstack((u_aspect_values, i_aspect_values))

                combined_dict = dict(zip(aspect_ids, aspect_values))
                sorted_dict = dict(sorted(combined_dict.items(), key=lambda item: item[1], reverse=True))
                exp_dict = {id2aspect[key]: value for key, value in sorted_dict.items()}
                users.append(user_idx2id[i])
                items.append(item_idx2id[j])
                explanations.append(exp_dict)
        u_i_gt = pd.DataFrame({'user_id': users, 'item_id': items, 'explanations':explanations})
        return u_i_gt


    def _create_ground_truth_limer(self):
        """ Generate ground truth explanations for each user-item pair
        
        Returns
        -------
        ground_truth explanation for each user,item pair [user_id, item_id, explanation] 
                    {'genre':1} for genre that is mentioned in item feature
        """
        def extract_gt(x,item=True):
            if item==True:
                temp = pd.Series(self.model.train_set.item_features[self.model.train_set.item_features['item_id'] == x]['feature'])
            else:
                temp = pd.Series(self.model.train_set.user_features[self.model.train_set.user_features['user_id'] == x]['feature'])
            exp = {f: 1 for f in temp}
            return exp
        
        def combined_exp(x):
            return {**x['exp_1'], **x['exp_2']}
        
        rec_df = self._create_recommendations()

        if self.item_f == True and self.user_f == True:
            rec_df['item_idx'] = rec_df['item_id'].apply(lambda x: str(self.model.train_set.iid_map[x]))
            rec_df['exp_1'] = rec_df['item_idx'].apply(lambda x: extract_gt(x,item=True))
            rec_df['user_idx'] = rec_df['user_id'].apply(lambda x: str(self.model.train_set.uid_map[x]))
            rec_df['exp_2'] = rec_df['user_idx'].apply(lambda x: extract_gt(x,item=False))
            rec_df['explanations'] = rec_df[['exp_1','exp_2']].apply(combined_exp, axis=1)
        
        elif self.item_f == True:
            rec_df['item_idx'] = rec_df['item_id'].apply(lambda x: str(self.model.train_set.iid_map[x]))
            rec_df['explanations'] = rec_df['item_idx'].apply(lambda x: extract_gt(x,item=True))
        else:
            rec_df['user_idx'] = rec_df['user_id'].apply(lambda x: str(self.model.train_set.uid_map[x]))
            rec_df['explanations'] = rec_df['user_idx'].apply(lambda x: extract_gt(x,item=False))
        
        rec_df = rec_df[['user_id', 'item_id', 'explanations']]
        return rec_df

    def compute(self, model, explainer, dataset=None):
        """ Compute the precision, recall, and f1 score for the explanations

        Parameters
        ----------
        model: object
            The recommendation model to evaluate.
        explainer: object
            The explainer model to evaluate.
        dataset: object, optional, default: None    
            The dataset to evaluate the model and explainer on. If None, the training set of the model will be used.
            
        Returns
        -------
        [precision, recall, f1]: [float, float, float]
            average values for all (user, item) pairs' ground truth (reviews) with predicted explanations
        [precision_list, recall_list, f1_list]: [list, list, list]
            list of precision, recall, and f1 values for all (user, item) pairs
        """
        
        
        self.model = model
        self.explainer = explainer
        self.dataset = dataset
        if dataset is None:
            self.dataset = model.train_set
        self.explainer = explainer
        if self.explainer.name == "LIMERS":
            self.item_f = False if self.model.train_set.item_features.empty else True
            self.user_f = False if self.model.train_set.user_features.empty else True

        u_i_gt = None
        if self.model.name == 'MTER':
            u_i_gt = self._create_ground_truth_sentiment()
        elif self.model.name == 'EFM':
            u_i_gt = self._creat_grount_truth_efm()
        elif self.model.name in ['fm_regressor']:
            u_i_gt = self._create_ground_truth_limer()
        else:
            raise ValueError("Model not supported.")
        
        cdef list precision_list = []
        cdef list recall_list = []
        cdef list f1_list = []
        cdef int total = u_i_gt.shape[0]
        cdef int interval = int(total * 0.1)
        cdef int i
        cdef double p, r
        with tqdm(total=total, desc="Start evaluation... ") as pbar:
            for i, row in u_i_gt.iterrows():
                features_gt = set(row['explanations'].keys())
                if len(features_gt) != 0:
                    features_pred = set(self.explainer.explain_one_recommendation_to_user(row['user_id'], row['item_id'], num_features=self.feature_k).keys())
                    if self.model.name == 'fm_regressor':
                        features_pred = [x[:-4] if x[-2:] == '_f' else x for x in list(features_pred)]
                    p = 1.0*len(features_gt.intersection(features_pred)) / len(features_pred)
                    r = 1.0*len(features_gt.intersection(features_pred)) / len(features_gt)
                    if (p + r) == 0:
                        continue
                    precision_list.append(p)
                    recall_list.append(r)
                    f1_list.append(2 * p * r / (p + r))
                    
                    if i % interval == 0:
                        pbar.update(interval)
        precision = np.mean(precision_list)
        recall = np.mean(recall_list)
        ff1 = np.mean(f1_list)
        
        return [precision, recall, ff1], [precision_list, recall_list, f1_list]

    def _create_recommendations(self):
        """create recommendations for all users available in the dataset"""
        print("Started creating recommendations...")
        users = [k for k in self.dataset.uid_map.keys()] 
        rec_df = self.model.recommend_to_multiple_users(users, self.rec_k) ##to be updated
        print("Finished creating recommendations...")
        return rec_df[['user_id', 'item_id']]
