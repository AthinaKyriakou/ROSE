from ..metrics import Metrics
class DIV(Metrics):
    """ Feature Diversity: the lower the better"""

    def __init__(self,name = "diversity", feature_k=10):
        super().__init__(name=name)

    def compute(self, explanations):
        """
        Perform explainer evaluation
        args:
            explanations (np.ndarray): explanations(without feature score), [['uid', 'iid', [f1, f2, ...]], ['uid', 'iid', [f2, f5, ...]],  ...]
        Return:
            fd: feature diversity, calculate the common features of each two explanations
        Reference:
        Lei Li, Yongfeng Zhang, and Li Chen. 2020. Generate Neural Template Explanations for Recommendation. 
        In Proceedings of the 29th ACM International Conference on Information & Knowledge Management (CIKM '20). 
        Association for Computing Machinery, New York, NY, USA, 755-764. https://doi.org/10.1145/3340531.3411992
        """
        #cdef double num_common_keys = 0.0
        cdef int N = len(explanations)
        cdef list explanation_sets = []
        cdef int i, j
        cdef set keys_i, keys_j

        if isinstance(explanations[0][2], dict):
            explanation_sets = [item[2].keys() for item in explanations]
        else:
            explanation_sets = [item[2] for item in explanations]
        fd_list = []
        for i in range(N - 1):
            for j in range(i + 1, N):
                keys_i = set(explanation_sets[i])
                keys_j = set(explanation_sets[j])
                length = max(len(keys_i), len(keys_j))
                if length <= 0:
                    fd_list.append(0)
                else:
                    fd_list.append(len(keys_i.intersection(keys_j))/length)
                #num_common_keys += len(keys_i.intersection(keys_j))
        sum_fd = sum(fd_list)
        fd = 2 * sum_fd /( N * (N - 1.0))

        return fd, fd_list
