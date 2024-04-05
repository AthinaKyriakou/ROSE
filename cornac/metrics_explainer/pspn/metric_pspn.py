import pandas as pd
import copy
from tqdm.auto import tqdm
import numpy as np
import numpy as np

from ..metrics import Metrics

class PSPNFNS(Metrics):
    """probability of sufficiency, probability of necessity and the harmonic mean"""
    def __init__(self, name="PSPNFNS", rec_k=10, feature_k=10, num_threads=0):
        
        Metrics.__init__(self, name=name, rec_k=rec_k, feature_k=feature_k)
        

    def compute(self, model, explainer, explanations):
        """
        perform explainer evaluation
        args:
            model: recommender model
            explainer: explanation method
            explanation: [user, item, explanation] 
        return: 
            tuple of (pn, ps, fns) and (pn_dist, ps_dist, fns_dist)
            ps: probability of sufficiency
            pn: probability of necessity
            fns: harmonic mean of overall performance: fns = 2*ps*pn / (pn+ps)
            
        Reference:
            Tan, J., Xu, S., Ge, Y., Li, Y., Chen, X., &amp; Zhang, Y. (2021). 
            Counterfactual explainable recommendation. 
            Proceedings of the 30th ACM International Conference on Information &amp;amp; Knowledge Management. 
            https://doi.org/10.1145/3459637.3482420 
        """
            
        self.model = model
        self.explainer = explainer
        self.explanations = explanations
        
        if self.explainer.name == "LIMERS":
            self.item_f = False
            self.user_f = False
            if not self.model.train_set.item_features.empty:
                self.item_f = True
                self.item_features_original = copy.deepcopy(self.model.train_set.item_features)
            if not self.model.train_set.user_features.empty:
                self.user_f = True
                self.user_features_original = copy.deepcopy(self.model.train_set.user_features)

        pn_count = 0
        ps_count = 0
        total = self.explanations.shape[0]
        interval = int(total * 0.1)
        ps_dist = {user:0 for user in set([user for (user,_,_) in explanations])}
        pn_dist = {user:0 for user in set([user for (user,_,_) in explanations])}
        
        with tqdm(total=total, desc="Re-evaluate after features removal... ") as pbar:
            for i, (user, item, exp) in enumerate(self.explanations):
                # remove features mentioned in exp in item_feature_matrix
                if self.explainer.name == 'LIMERS':
                    rank_scores_pn = self._reevaluate_limers(exp, user, pn=True)
                    rank_scores_ps = self._reevaluate_limers(exp, user, pn=False)
                elif self.explainer.name in ['EFM_Exp', 'MTER_Exp']:
                    rank_scores_pn = self._reevaluate_sentiment(exp, user, pn=True)
                    rank_scores_ps = self._reevaluate_sentiment(exp, user, pn=False)
                else:
                    raise ValueError(f"Metric {self.name} does not support {self.explainer.name}")
                if item not in rank_scores_pn:
                    pn_count += 1
                    pn_dist[user] += 1
                if item in rank_scores_ps:
                    ps_count += 1
                    ps_dist[user] += 1

                if i % interval == 0:
                    pbar.update(interval)

        if len(self.explanations) != 0:
            pn = pn_count / len(self.explanations)
            pn_dist = [value/self.rec_k for value in pn_dist.values()]
            ps = ps_count / len(self.explanations)
            ps_dist = [value/self.rec_k for value in ps_dist.values()]
            if (pn + ps) != 0:
                fns = 2 * pn * ps / (pn + ps)
                fns_dist = [2 * pn_dist[i] * ps_dist[i] / (pn_dist[i] + ps_dist[i]) if (pn_dist[i] + ps_dist[i]) > 0 else 0 for i,_ in enumerate(pn_dist)]
            else:
                fns = 0
        else:
            pn = 0
            ps = 0
            fns = 0
        return [pn, ps, fns], [pn_dist, ps_dist, fns_dist]

    def _reevaluate_limers(self, exp, user, pn=True):
        """
        For limers explainers: re-evaluate recommendation list for user after modifying the feature input or item/user matrix
        args:
            exp: list containing feature/aspect/item from explanation model
            user: user of interest
            rec_k: number of recommended items in model.recommend()
            pn: if True follow modification for probability of necesity; if False fofllow modification for probability of sufficiency
        """
        if pn:
            if self.item_f == True and self.user_f == True:
                item_exp = [x[:-4] if x[-4:] == '_i_f' else x for x in exp]
                user_exp = [x[:-4] if x[-4:] == '_u_f' else x for x in exp]
                self.model.train_set.item_features = self.item_features_original[~self.item_features_original['feature'].isin(item_exp)]
                self.model.train_set.user_features = self.user_features_original[~self.user_features_original['feature'].isin(user_exp)]
            elif self.item_f == True:
                self.model.train_set.item_features = self.item_features_original[~self.item_features_original['feature'].isin(exp)]
            else:
                self.model.train_set.user_features = self.user_features_original[~self.user_features_original['feature'].isin(exp)]

        else:
            if self.item_f == True and self.user_f == True:
                item_exp = [x[:-4] if x[-4:] == '_i_f' else x for x in exp]
                user_exp = [x[:-4] if x[-4:] == '_u_f' else x for x in exp]
                self.model.train_set.item_features = self.item_features_original[self.item_features_original['feature'].isin(item_exp)] 
                self.model.train_set.user_features = self.user_features_original[self.user_features_original['feature'].isin(user_exp)]   
            elif self.item_f == True:
                self.model.train_set.item_features = self.item_features_original[self.item_features_original['feature'].isin(exp)]  
            else:
                self.model.train_set.user_features = self.user_features_original[self.user_features_original['feature'].isin(exp)] 

        rank_scores = self.model.recommend_to_multiple_users([user], self.rec_k)['item_id'].to_list()
        
        if not self.model.train_set.item_features.empty:
            self.model.train_set.item_features = self.item_features_original
        if not self.model.train_set.user_features.empty:
            self.model.train_set.user_features = self.user_features_original
        return rank_scores
    

    def _reevaluate_sentiment(self, exp, user, pn=True):
        """
        For sentiment explainers: re-evaluate recommendation list for user after modifying the feature input or item/user matrix
        args:
            exp: list containing feature/aspect/item from explanation model
            user: user of interest
            pn: if True follow modification for probability of necesity; if False fofllow modification for probability of sufficiency
        """
        model_copy = copy.deepcopy(self.model)
        features = exp
        indices = [self.model.train_set.sentiment.aspect_id_map[f] for f in features]
        if self.model.name == 'EFM':
            if pn:
                self.model.V = np.delete(self.model.V, indices, axis=0)
            else:
                self.model.V = self.model.V[indices]
            rank_scores = self.model.recommend_to_multiple_users([user], self.rec_k)['item_id'].to_list()
            self.model.V = model_copy.V
            return rank_scores
        elif self.model.name == 'MTER':
            if pn:
                self.model.A = np.delete(self.model.A, indices, axis=0)
            else:
                self.model.A = self.model.A[indices]
            rank_scores = self.model.recommend_to_multiple_users([user], self.rec_k)['item_id'].to_list()
            self.model.A = model_copy.A
            return rank_scores
        else:
            pass

    