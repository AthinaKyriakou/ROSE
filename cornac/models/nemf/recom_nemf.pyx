import scipy
import numpy as np
import pandas as pd
from tqdm.auto import trange

cimport cython
from cython.parallel import prange
from cython cimport floating, integral
from libcpp cimport bool
from libc.math cimport fabs, signbit
cimport numpy as cnp

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import pairwise_distances
import multiprocessing

from cornac.models import Recommender
from cornac.utils import get_rng, fast_dot
from cornac.utils.init_utils import normal, zeros
from cornac.exception import ScoreException


class NEMF(Recommender):
    """ Novel and Explainable Matrix Factorisation.
    
    Parameters
    ----------
    k: int, optional, default: 10
        The dimension of the latent factors.
    knn_num: int, optional, default: 10
        The number of nearest neighbors to be used for the explanation.
    knn_threshold: float, optional, default: 0.0
        The threshold for the edge weight matrix between user-item pairs.
    positive_rating_threshold: float, optional, default: 0.0
        The threshold for the positive ratings.
    max_iter: int, optional, default: 100
        Maximum number of iterations or the number of epochs for SGD.
    learning_rate: float, optional, default: 0.001
        The learning rate.
    lambda_reg: float, optional, default: 0.01
        The lambda value used for regularization.
    explain_reg: float, optional, default: 0.1
        The lambda value used for regularization of the explanation.
    novel_reg: float, optional, default: 0.1
        The delta value used for regularization of the novel matrix.
    use_bias: boolean, optional, default: True
        When True, user, item, and global biases are used.
    early_stop: boolean, optional, default: False
        When True, delta loss will be checked after each iteration to stop learning earlier.
    num_threads: int, optional, default: 0
        Number of parallel threads for training. If num_threads=0, all CPU cores will be utilized.
        If seed is not None, num_threads=1 to remove randomness from parallelization.
    trainable: boolean, optional, default: True
        When False, the model will not be re-trained, and input of pre-trained parameters are required.
    verbose: boolean, optional, default: True
        When True, running logs are displayed.
    init_params: dictionary, optional, default: None
        Initial parameters, e.g., init_params = {'U': user_factors, 'V': item_factors,
        'Bu': user_biases, 'Bi': item_biases}
    seed: int, optional, default: None
        Random seed for weight initialization.
        If specified, training will take longer because of single-thread (no parallelization).
    distance_metric: string, optional, default: 'euclidean'
        The distance metric used for computing the novel matrix.
    sim_filter_zeros: boolean, optional, default: True
        When True, the similarity matrix will be computed by filtering out the zero values.
        When False, the similarity matrix will be computed by keeping the zero values.
    
    References
    ----------
    [1] L. Coba, P. Symeonidis, and M. Zanker, “Personalised novel and explainable matrix factorisation,” 
    Data & Knowledge Engineering, vol. 122, pp. 142-158, Jul. 2019, doi: 10.1016/j.datak.2019.06.003.
    """

    def __init__(
        self,
        name='NEMF',
        k=10,
        knn_num=10,
        knn_threshold=0.0,
        positive_rating_threshold=0.0,
        max_iter=20,
        learning_rate=0.001,
        lambda_reg=0.01,
        explain_reg=0.1,
        novel_reg=0.1,
        use_bias=True,
        early_stop=False,
        num_threads=0,
        trainable=True,
        verbose=False,
        init_params=None,
        seed=None,
        distance_metric='euclidean',
        sim_filter_zeros=True
    ):
        # super().__init__(name=name, k=k, max_iter=max_iter, learning_rate=learning_rate, lambda_reg=lambda_reg, use_bias=use_bias,
        #                  early_stop=early_stop, num_threads=num_threads, trainable=trainable, verbose=verbose, init_params=init_params, seed=seed)
        super().__init__(name=name, trainable=trainable, verbose=verbose)
        self.k = k
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.lambda_reg = lambda_reg
        self.novel_reg = novel_reg
        self.use_bias = use_bias
        self.seed = seed
        self.distance_metric = distance_metric
        self.early_stop = early_stop
        self.sim_filter_zeros = sim_filter_zeros
        


        if seed is not None:
            self.num_threads = 1
        elif num_threads > 0 and num_threads < multiprocessing.cpu_count():
            self.num_threads = num_threads
        else:
            self.num_threads = multiprocessing.cpu_count()

        # Init params if provided
        self.init_params = {} if init_params is None else init_params
        self.u_factors = self.init_params.get('U', None)
        self.i_factors = self.init_params.get('V', None)
        self.u_biases = self.init_params.get('Bu', None)
        self.i_biases = self.init_params.get('Bi', None)
        self.global_mean = 0.0

        self.knn_num = knn_num
        self.knn_threshold = knn_threshold
        self.positive_rating_threshold = positive_rating_threshold
        self.explain_reg = explain_reg
        self.edge_weight_matrix = None
        self.novel_matrix = None
        self.sim_users = {}

    def _init(self):
        rng = get_rng(self.seed)
        n_users, n_items = self.train_set.num_users, self.train_set.num_items

        if self.u_factors is None:
            self.u_factors = normal(
                [n_users, self.k], std=0.01, random_state=rng)
        if self.i_factors is None:
            self.i_factors = normal(
                [n_items, self.k], std=0.01, random_state=rng)

        self.u_biases = zeros(
            n_users) if self.u_biases is None else self.u_biases
        self.i_biases = zeros(
            n_items) if self.i_biases is None else self.i_biases
        self.global_mean = self.train_set.global_mean if self.use_bias else 0.0

    def fit(self, train_set, val_set=None):
        """Fit the model to observations.

        Parameters
        ----------
        train_set: :obj:`cornac.data.Dataset`, required
            User-Item preference data as well as additional modalities.

        val_set: :obj:`cornac.data.Dataset`, optional, default: None
            User-Item preference data for model selection purposes (e.g., early stopping).

        Returns
        -------
        self : object
        """
        Recommender.fit(self, train_set, val_set)

        self._init()

        if self.trainable:
            (rid, cid, val) = train_set.uir_tuple
            self._fit_sgd(rid, cid, val.astype(np.float32),
                          self.u_factors, self.i_factors,
                          self.u_biases, self.i_biases)

        return self

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _fit_sgd(self, integral[:] rid, integral[:] cid, floating[:] val,
                 floating[:, :] U, floating[:, :] V, floating[:] Bu, floating[:] Bi):
        """Fit the model parameters (U, V, Bu, Bi) with SGD
        """
        cdef:
            long num_users = self.train_set.num_users
            long num_items = self.train_set.num_items
            long num_ratings = val.shape[0]
            int num_factors = self.k
            int max_iter = self.max_iter
            int num_threads = self.num_threads

            floating reg = self.lambda_reg
            floating expl_reg = self.explain_reg
            floating novel_reg = self.novel_reg
            floating mu = self.global_mean

            bool use_bias = self.use_bias
            bool early_stop = self.early_stop
            bool verbose = self.verbose

            floating lr = self.learning_rate
            floating loss = 0
            floating last_loss = 0
            floating r, r_pred, error, u_f, i_f, delta_loss, temp
            integral u, i, f, j

            floating * user
            floating * item

        print('Start compute edge weight matrix...')
        self.compute_edge_weight_matrix()
        cdef double[:, :] edge_weight_matrix = self.edge_weight_matrix
        print('Start compute novel matrix...')
        self.compute_novel_matrix()
        cdef double[:, :] novel_matrix = self.novel_matrix
        print('Matrix computation finished!')

        progress = trange(max_iter, disable=not self.verbose)
        for epoch in progress:
            last_loss = loss
            loss = 0

            for j in prange(num_ratings, nogil=True, schedule='static', num_threads=num_threads):
                u, i, r = rid[j], cid[j], val[j]
                user, item = &U[u, 0], &V[i, 0]

                # predict rating
                r_pred = mu + Bu[u] + Bi[i]
                for f in range(num_factors):
                    r_pred += user[f] * item[f]

                # error = r - r_pred
                loss += (r - r_pred) * (r - r_pred)

                for f in range(num_factors):
                    u_f, i_f = user[f], item[f]
                    user[f] += lr * ((2.0 * (r - r_pred) * i_f - reg * u_f) - ((signbit(i_f - u_f) * (-2.0) + 1.0) * (expl_reg * edge_weight_matrix[u, i] + novel_reg * novel_matrix[u, i])))
                    item[f] += lr * ((2.0 * (r - r_pred) * u_f - reg * i_f) - (1.0 * (signbit(u_f - i_f) * (-2.0) + 1.0) * (expl_reg * edge_weight_matrix[u, i] + novel_reg * novel_matrix[u, i])))


                # delta_i = 2.0 * error * user - reg * item
                # temp = 1.0 * np.sign(user - item) *  (expl_reg * edge_weight_matrix[u, i] + novel_reg * novel_matrix[u, i])
                # delta_i -= temp
                # item += lr * delta_i

                # update biases
                if use_bias:
                    Bu[u] += lr * ((2.0 * (r - r_pred) - reg * Bu[u]) - (signbit(Bu[u] - Bi[i]) * (-2.0) + 1.0) * (expl_reg * edge_weight_matrix[u, i] + novel_reg * novel_matrix[u, i]))
                    Bi[i] += lr * ((2.0 * (r - r_pred) - reg * Bi[i]) - (signbit(Bi[i] - Bu[u]) * (-2.0) + 1.0) * (expl_reg * edge_weight_matrix[u, i] + novel_reg * novel_matrix[u, i]))

            loss = 0.5 * loss
            progress.update(1)
            progress.set_postfix({"loss": "%.2f" % loss})

            delta_loss = loss - last_loss
            if early_stop and fabs(delta_loss) < 1e-5:
                if verbose:
                    print('Early stopping, delta_loss = %.4f' % delta_loss)
                break
        progress.close()

        if verbose:
            print('Optimization finished!')

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _c_get_similarities_filter_zeros(self, ui_matrix):
        """
        Compute the similarities between users
        """
        cdef:
            long num_users = self.train_set.num_users
            int i, j, k
            int num_threads = self.num_threads
            int * a
            int * b
            double[:, :] similarities = np.zeros((num_users, num_users))
            double[:, :] sim = np.zeros((num_users, num_users))
            double[:, :] lena = np.zeros((num_users, num_users))
            double[:, :] lenb = np.zeros((num_users, num_users))
            int len_matrix = ui_matrix.shape[1]
            int[:, :] matrix = ui_matrix

        for i in prange(num_users, nogil=True, schedule='static', num_threads=num_threads):
            for j in range(num_users):
                if i == j:
                    continue
                a = &matrix[i, 0]
                b = &matrix[j, 0]
                for k in range(len_matrix):
                    if a[k] == 0 or b[k] == 0:
                        continue
                    sim[i][j] += a[k] * b[k]
                    lena[i][j] += a[k] * a[k]
                    lenb[i][j] += b[k] * b[k]

        for i in prange(num_users, nogil=True, schedule='static', num_threads=num_threads):
            for j in range(num_users):
                if i == j:
                    similarities[i][j] = -2
                    continue
                if lena[i][j] == 0 or lenb[i][j] == 0 or sim[i][j] == 0:
                    similarities[i][j] = 0
                    continue
                similarities[i][j] = 1.0 * sim[i][j]
                similarities[i][j] /= (1.0 * lena[i][j]) ** 0.5
                similarities[i][j] /= (1.0 * lenb[i][j]) ** 0.5
                
        return np.asarray(similarities)


    def compute_edge_weight_matrix(self):
        """Compute the edge weight matrix between user-item pairs of the model.
        """
        ds = self.train_set.matrix  # this is the user-item interaction matrix in CSR sparse format
        if self.sim_filter_zeros:
            sim_matrix = self._c_get_similarities_filter_zeros(ds.toarray().astype(np.int32))
        else:
            sim_matrix = cosine_similarity(ds)

        self.edge_weight_matrix = np.zeros((self.train_set.num_users, self.train_set.num_items))

        num_users = self.train_set.num_users
        num_items = self.train_set.num_items

        for i in range(num_users):
            sim_matrix[i][i] = -2
            self.sim_users[i] = np.argsort(sim_matrix[i])[::-1][:self.knn_num]

        self.edge_weight_matrix = np.zeros((num_users, num_items))

        filted_positive_rating_df = pd.DataFrame(
            np.array(self.train_set.uir_tuple).T, columns=['user', 'item', 'rating'])
        filted_positive_rating_df['user'] = filted_positive_rating_df['user'].astype(
            int)
        filted_positive_rating_df['item'] = filted_positive_rating_df['item'].astype(
            int)
        filted_positive_rating_df = filted_positive_rating_df[
            filted_positive_rating_df['rating'] >= self.positive_rating_threshold]
        for i in range(num_users):
            sim_users_i = self.sim_users[i]
            rated_items_by_sim_users = filted_positive_rating_df[filted_positive_rating_df['user'].isin(
                sim_users_i)]
            sim_ratings_items = rated_items_by_sim_users.groupby('item')
            sim_ratings_sum = sim_ratings_items['rating'].sum()

            self.edge_weight_matrix[i][sim_ratings_sum.index] = sim_ratings_sum.values

        self.edge_weight_matrix = MinMaxScaler().fit_transform(self.edge_weight_matrix)
        self.edge_weight_matrix[self.edge_weight_matrix <=
                                self.knn_threshold] = 0
        return self.edge_weight_matrix

    def compute_novel_matrix(self):
        """Compute the novel matrix of the model using distance-based model.
        """
        ds = self.train_set.csc_matrix.T  # this is the user-item interaction matrix in CSC sparse format
        distance_between_items = pairwise_distances(ds, metric=self.distance_metric)

        num_users = self.train_set.num_users
        num_items = self.train_set.num_items

        df = pd.DataFrame(np.array(self.train_set.uir_tuple).T, columns=['user', 'item', 'rating'])
        df['user'] = df['user'].astype(int)
        df['item'] = df['item'].astype(int)

        novel_matrix = np.zeros((num_users, num_items))
        for i in range(num_users):
            novel_matrix[i] = np.mean(distance_between_items[df[df['user'] == i]['item']], axis=0)
        
        novel_matrix = MinMaxScaler().fit_transform(novel_matrix)
        self.novel_matrix = novel_matrix
        return novel_matrix


    def score(self, user_idx, item_idx=None):
        """Predict the scores/ratings of a user for an item.

        Parameters
        ----------
        user_idx: int, required
            The index of the user for whom to perform score prediction.

        item_idx: int, optional, default: None
            The index of the item for which to perform score prediction.
            If None, scores for all known items will be returned.

        Returns
        -------
        res : A scalar or a Numpy array
            Relative scores that the user gives to the item or to all known items

        """
        unk_user = self.is_unknown_user(user_idx)

        if item_idx is None:
            known_item_scores = np.add(self.i_biases, self.global_mean)
            if not unk_user:
                known_item_scores = np.add(
                    known_item_scores, self.u_biases[user_idx])
                fast_dot(self.u_factors[user_idx],
                         self.i_factors, known_item_scores)
            return known_item_scores
        else:
            unk_item = self.is_unknown_item(item_idx)
            if self.use_bias:
                item_score = self.global_mean
                if not unk_user:
                    item_score += self.u_biases[user_idx]
                if not unk_item:
                    item_score += self.i_biases[item_idx]
                if not unk_user and not unk_item:
                    item_score += np.dot(self.u_factors[user_idx],
                                         self.i_factors[item_idx])
            else:
                if unk_user or unk_item:
                    raise ScoreException(
                        "Can't make score prediction for (user_id=%d, item_id=%d)" % (user_idx, item_idx))
                item_score = np.dot(
                    self.u_factors[user_idx], self.i_factors[item_idx])
            return item_score

    def user_embedding(self):
        return self.u_factors

    def item_embedding(self):
        return self.i_factors
