import scipy
import numpy as np
import pandas as pd
from tqdm.auto import trange

import multiprocessing

from cornac.models import Recommender
from cornac.utils import get_rng, fast_dot
from cornac.utils.init_utils import normal, zeros
from cornac.exception import ScoreException

from implicit.als import AlternatingLeastSquares as als

class ALS(Recommender):
    """Alternating Least Squares of Matrix Factorization.

    Parameters
    ----------
    k: int, optional, default: 10
        The dimension of the latent factors.

    max_iter: int, optional, default: 100
        Maximum number of iterations or the number of epochs for SGD.

    lambda_reg: float, optional, default: 0.001
        The lambda value used for regularization.

    alpha: float, optional, default: 1.0
        The rate of confidence increase

    num_threads: int, optional, default: 0
        Number of parallel threads for training. If num_threads=0, all CPU cores will be utilized.
        If seed is not None, num_threads=1 to remove randomness from parallelization.

    trainable: boolean, optional, default: True
        When False, the model will not be re-trained, and input of pre-trained parameters are required.

    verbose: boolean, optional, default: True
        When True, running logs are displayed.

    init_params: dictionary, optional, default: None
        Initial parameters, e.g., init_params = {'U': user_factors, 'V': item_factors}

    seed: int, optional, default: None
        Random seed for weight initialization.
        If specified, training will take longer because of single-thread (no parallelization).

    References
    ----------
    [1] Y. Hu, Y. Koren, and C. Volinsky, “Collaborative Filtering for Implicit Feedback Datasets,” in 2008 Eighth IEEE International Conference on Data Mining, Pisa, Italy: IEEE, Dec. 2008, pp. 263-272. doi: 10.1109/ICDM.2008.22.
    
    [2] implicit library: https://pypi.org/project/implicit/
    """
    
    

    def __init__(
        self, 
        name='ALS', 
        k=10, 
        max_iter=20, 
        # learning_rate=0.01, 
        lambda_reg=0.02, 
        alpha = 1.0,
        num_threads=0, 
        trainable=True, 
        verbose=False, 
        init_params=None, 
        seed=None
    ):
        super().__init__(name=name, trainable=trainable, verbose=verbose)
        self.k = k
        self.max_iter = max_iter
        self.alpha = alpha
        self.lambda_reg = lambda_reg
        self.seed = seed

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
        self.global_mean = 0.0

        self.Cui = None
        self.Ciu = None

    def _init(self):
        rng = get_rng(self.seed)
        n_users, n_items = self.train_set.num_users, self.train_set.num_items

        if self.u_factors is None:
            self.u_factors = normal([n_users, self.k], std=0.01, random_state=rng) 
        if self.i_factors is None:
            self.i_factors = normal([n_items, self.k], std=0.01, random_state=rng)

        self.global_mean = 0.0


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
            a = als(factors=self.k, 
            regularization=self.lambda_reg,
            alpha=self.alpha,
            iterations=self.max_iter,
            calculate_training_loss=True,
            num_threads=self.num_threads,
            random_state=self.seed)
            a.fit(self.train_set.matrix)
            self.u_factors = np.array(a.user_factors, dtype=np.float64)
            self.i_factors = np.array(a.item_factors, dtype=np.float64)

        return self


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
            known_item_scores = np.zeros(self.train_set.num_items)
            if not unk_user:
                fast_dot(self.u_factors[user_idx], self.i_factors, known_item_scores)
            return known_item_scores
        else:
            unk_item = self.is_unknown_item(item_idx)
            if unk_user or unk_item:
                raise ScoreException("Can't make score prediction for (user_id=%d, item_id=%d)" % (user_idx, item_idx))
            item_score = np.dot(self.u_factors[user_idx], self.i_factors[item_idx])
            return item_score

