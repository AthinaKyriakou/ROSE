import numpy as np
import pandas as pd
cimport numpy as np
cimport cython
from cython.parallel import prange
import multiprocessing
from tqdm import trange

from ..metrics import Metrics

class EnDCG(Metrics):
    """ Explainable normalised Discounted Cumulative Gain
    Parameters
    ----------
    rec_k: int, optional, default: 10
        Number of recommendations for each user.
    feature_k: int, optional, default: 10
        Number of elements in one explanation.
    num_threads: int, optional, default: 0
        Number of parallel threads for training. If num_threads=0, all CPU cores will be utilized.
        If seed is not None, num_threads=1 to remove randomness from parallelization.
    name: str, optional, default: 'EnDCG'
    """
    def __init__(self,
             rec_k=10,
             feature_k=10,
             num_threads=0,
             name='EnDCG'):

        super().__init__(name=name, rec_k=rec_k, feature_k=feature_k)

        if num_threads > 0 and num_threads < multiprocessing.cpu_count():
            self.num_threads = num_threads
        else:
            self.num_threads = multiprocessing.cpu_count()
 
        self.N = rec_k

        

    def compute(self,
            recommender,
            recommendations=None):
        """
        Main function to compute the expected average EnDCG for all users.
        Parameters
        ----------
        recommender: instance of a recommender model
            Trained recommender model.
        recommendations: pd.DataFrame, optional, default: None
            Recommendations for all users. If None, the recommendations will be computed on the fly.
        """
        self.model = recommender      
        #if dataset is None and hasattr(model, 'train_set'):
        #    dataset = model.train_set
        #elif dataset is None and not hasattr(model, 'train_set'):
        #    print('Please provide a dataset or train model first!')
        #    return
        if not hasattr(recommender, 'train_set'):
            # print('Please train model first!')
            # return
            raise AttributeError("The model is not trained yet.")
        self.dataset = recommender.train_set
        
        # check if the model has edge_weight_matrix
        if not hasattr(self.model, 'edge_weight_matrix'):
            # print('The model can not be use for MEP!')
            # return
            raise NotImplementedError("The model does not have edge_weight_matrix.")
        
        if self.model.edge_weight_matrix is None:
            # print('Your model is not trained yet. Here we will train it for you!')
            # self.model.fit(self.dataset)
            # print('Please train model first!')
            # return
            raise AttributeError("The model is not trained yet.")
            
        self.E = self.model.edge_weight_matrix
        self.U = self.dataset.num_users
        self.recommendations = recommendations
        self.E_DCG_u = None
        self.N = self.rec_k
        
        self._check_recommendations(self.N)
        EnDCG = self._c_compute_E_nDCG()
        return EnDCG, self.E_DCG_u

        
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _c_compute_E_nDCG(self):
        """
        Use cython to compute the expected average EnDCG for all users.
        Only called by compute_E_nDCG().
        """
        log2list = np.log2(np.arange(1, self.N + 2))
        log2list[0] = 1

        cdef double [:] log2list_c = log2list
        cdef int u, i, j, k
        cdef int count = 0
        cdef int last = -1
        cdef int [:, :] recommendations = np.array(self.recommendations, dtype=np.int32)
        cdef double [:, :] E = self.E        
        cdef double E_nDCG = 0.0
        cdef int num_threads = self.num_threads
        cdef int len_rec = len(recommendations)
        cdef double E_IDCG = self._compute_E_IDCG()
        cdef int U = self.U
        cdef int N = self.N
        cdef double [:] E_DCG_u = np.zeros(U, dtype=np.float64)


        for j in prange(U, nogil=True, num_threads=num_threads):
            u = recommendations[j*N, 0]
            for k in range(N):
                i = recommendations[j*N + k, 1]
                E_DCG_u[u] += E[u, i] / log2list_c[k]
            E_DCG_u[u] /= E_IDCG
            E_nDCG += E_DCG_u[u]
            
        self.E_DCG_u = np.array(E_DCG_u)
        return E_nDCG / U
    

    def _compute_E_IDCG(self):
        """
        Compute the expected IDCG with E_max.
        Only called by _c_compute_E_nDCG().
        """
        E_max = np.max(self.E)
        E_IDCG = E_max
        for i in range(1, self.N):
            E_IDCG += E_max / np.log2(i + 1)
        return E_IDCG


    def _check_recommendations(self, N=0):
        """
        Cold start for recommendations. If the recommendations aren't provided, we need to generate them. 
        But it's really slow. 
        If recommendations are provided in users' ids not index, map the ids to index.
        Only called by compute_E_nDCG().
        """
        if self.recommendations is None or self.N != N:
            print('Computing recommendations for all users...')
            r4au = self.model.recommend_to_multiple_users(list(self.dataset.user_ids), N)
            userid2idx = self.dataset.uid_map
            itemid2idx = self.dataset.iid_map
            self.recommendations = [(userid2idx[r4au['user_id'].values[i]], itemid2idx[r4au['item_id'].values[i]]) for i in range(len(r4au))]
            self.N = N
            print('Done!')
        elif self.recommendations['user_id'].values[0] in self.dataset.uid_map.keys():
            userid2idx = self.dataset.uid_map
            itemid2idx = self.dataset.iid_map
            r4au = self.recommendations
            new_rec = [(userid2idx[r4au['user_id'].values[i]], itemid2idx[r4au['item_id'].values[i]]) for i in range(len(r4au))]
            self.recommendations = new_rec
