import numpy as np
cimport numpy as np
from ..metric_exp import Metric_Exp


class Metric_Exp_RA(Metric_Exp):
    """
        Rank Agreement: computes the fraction of features that are 
        not only common between the sets of top-k features of two explanations, 
        but also have the same position in the respective rank orders.
        a stricter metric than feature agreement since it also considers the ordering of the top-k features.
        
        Parameters
        ----------
        name: str, default='Metric_Exp_RA'
        feature_k: int, default=10
            Number of top features to consider for the metric.
            
        Reference
        ---------
        [1] Krishna, Satyapriya & Han, etc. (2022). https://doi.org/10.48550/arXiv.2202.01602
            The Disagreement Problem in Explainable Machine Learning: A Practitioner's Perspective.
    """
    def __init__(self, name = 'Metric_Exp_RA', feature_k = 10):
        super().__init__(name=name, feature_k=feature_k)
    
    def compute(self, exp_a, exp_b):
        """Compute the Rank Agreement for two explanations.
        
        Parameters
        ----------
        exp_a: np.ndarray
            The first explanation.
        exp_b: np.ndarray   
            The second explanation.
        
        Returns
        -------
        ra: float
            Rank Agreement value.
        ra_list: list
            List of Rank Agreement values for each instance.
        """
        
        is_dictionary = isinstance(exp_a[0][2], dict)
        cdef double ra = 0.0
        cdef double ra_avg=0.0
        # filter out the same row in two array
        
        #exp_a = np.array([row for row in exp_a if any(np.all(row[:1] == r[:1]) for r in exp_b)])
        #exp_b = np.array([r for row in exp_a if any(np.all(row[:1] == r[:1]) for r in exp_b)])
        new_exp_a = []
        new_exp_b = []
        for row in exp_a:
            for r in exp_b:
                if len(r[2]) >0 and  row[:1] == r[:1]:
                    new_exp_a.append(row)
                    new_exp_b.append(r)
        exp_a = np.array(new_exp_a)
        exp_b = np.array(new_exp_b)
        
        cdef int N = len(exp_a)
        cdef int i
        cdef int common_num=0
        ra_list = []
        if is_dictionary:
            for i in range(len(exp_a)):
                rank_list_a = list(exp_a[i][2].items())
                rank_list_b = list(exp_b[i][2].items())
                min_num = min(len(rank_list_a), len(rank_list_b))
                max_num = max(len(rank_list_a), len(rank_list_b))
                common_num=0
                for j in range(min_num):
                    if rank_list_a[j]==rank_list_b[j]:
                        common_num +=1
                if max_num > 0:
                    ra_list.append(1.0*common_num/max_num)
            ra_sum = sum(ra_list)
            ra_avg= 1.0*ra_sum/N if N>0 else 0
            return ra_avg, ra_list
            
        else:
            for i in range(len(exp_a)):
                rank_list_a = exp_a[i][2]
                rank_list_b = exp_b[i][2]
                common_num=0
                min_num = min(len(rank_list_a), len(rank_list_b))
                max_num = min(len(rank_list_a), len(rank_list_b))
                for j in range(min_num):
                    if rank_list_a[j]==rank_list_b[j]:
                        common_num +=1
                if max_num > 0:
                    ra_list.append(1.0*common_num/max_num)
            ra_sum = sum(ra_list)
            ra_avg = 1.0* ra_sum /N if N>0 else 0
            return ra_avg, ra_list
