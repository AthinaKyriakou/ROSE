import numpy as np
cimport numpy as np
from ..metrics import Metrics
class FA(Metrics):
    """ 
    Feature Agreement: computes the fraction of common features between the sets of top-k features of two explanations.
    """
    def __init__(self, name="FA", feature_k=10):
        super().__init__(name=name, feature_k=feature_k)
        
    def compute(self, exp_a, exp_b):
        """Compute the feature agreement between explanations of two different explainer.
        FA = number of common features of explanations from different explainers / rec_k
        Args:
            exp_a (np.ndarray): explanations(without feature score) of explainer a, [['uid', 'iid', [f1, f2, ...]], ['uid', 'iid', [f2, f5, ...]],  ...]
            exp_b (np.ndarray): explanations(without feature score) of explainer b
        Returns:
            fa (float): value of feature agreement, the higher the better
        References:
            Krishna, Satyapriya & Han, etc. (2022). https://doi.org/10.48550/arXiv.2202.01602
            The Disagreement Problem in Explainable Machine Learning: A Practitioner's Perspective.
        """
        cdef double fa = 0
        cdef double fa_avg=0.0
        #self._check_format(exp_a, exp_b)
        is_dictionary = isinstance(exp_a[0][2], dict)
        #exp_a = [row for row in exp_a if any(np.all(row[:1] == r[:1]) for r in exp_b)]
        #exp_b = [r for row in exp_a if any(np.all(row[:1] == r[:1]) for r in exp_b)]
        new_exp_a = []
        new_exp_b = []
        for row in exp_a:
            for r in exp_b:
                if len(r[2]) >0 and row[:1] == r[:1]:
                    new_exp_a.append(row)
                    new_exp_b.append(r)
        exp_a = np.array(new_exp_a)
        exp_b = np.array(new_exp_b)
        cdef int N = len(exp_a)
        cdef int i
        cdef int common_num=0
        fa_list = []
        if is_dictionary:
            for i in range(len(exp_a)):
                rank_list_a = list(exp_a[i][2].items())
                rank_list_b = list(exp_b[i][2].items())
                min_num = min(len(rank_list_a), len(rank_list_b))
                max_num = max(len(rank_list_a), len(rank_list_b))
                common_num=0
                for item in rank_list_a:
                    if item in rank_list_b:
                        common_num+=1
                if max_num > 0:
                    fa_list.append(1.0*common_num/max_num)
            fa_sum = sum(fa_list)
            fa_avg= 1.0*fa_sum/N if N>0 else 0
            return fa_avg, fa_list
        else:
            for i in range(len(exp_a)):
                common_num=0
                min_num = 0
                max_num = 0
                for feature in exp_a[i][2]:
                    min_num = min(len(exp_a[i][2]), len(exp_b[i][2]))
                    max_num = max(len(exp_a[i][2]), len(exp_b[i][2]))
                    if feature in exp_b[i][2]:
                        common_num += 1
                if max_num > 0:
                    fa_list.append(1.0*common_num/max_num) 
            fa_sum = sum(fa_list)
            fa_avg = 1.0*fa_sum/N if N>0 else 0
            return fa_avg, fa_list
    
    def _check_format(self, exp_a, exp_b):
        if(exp_a.shape[0] != exp_b.shape[0]):
            raise ValueError(f"Error from {self.name}, the parameters must have the same shape.")
        if len(exp_a[0][2]) < self.feature_k or len(exp_b[0][2]) < self.feature_k:
            raise ValueError(f"Error from {self.name}, explanations have less than desired {self.feature_k} features.")
        else:
            return True