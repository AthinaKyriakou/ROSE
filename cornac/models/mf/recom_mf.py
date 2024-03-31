# Copyright 2018 The Cornac Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================


import multiprocessing

import numpy as np
import pandas as pd

from ..recommender import Recommender
from ..recommender import ANNMixin, MEASURE_DOT
from ...exception import ScoreException
from ...utils import fast_dot
from ...utils import get_rng
from ...utils.init_utils import normal, zeros


class MF(Recommender, ANNMixin):
    """Matrix Factorization.

    Parameters
    ----------
    k: int, optional, default: 10
        The dimension of the latent factors.

    backend: str, optional, default: 'cpu'
        Backend used for model training: cpu, pytorch
    
    optimizer: str, optional, default: 'sgd'
        Specify an optimizer: adagrad, adam, rmsprop, sgd. (ineffective if using CPU backend)
        
    max_iter: int, optional, default: 100
        Maximum number of iterations or the number of epochs for training.

    learning_rate: float, optional, default: 0.01
        The learning rate.
        
    batch_size: int, optional, default: 256
        Batch size (ineffective if using CPU backend).

    lambda_reg: float, optional, default: 0.001
        The lambda value used for regularization.
        
    dropout: float, optional, default: 0.0
        The dropout rate of embedding. (ineffective if using CPU backend)

    use_bias: boolean, optional, default: True
        When True, user, item, and global biases are used.

    early_stop: boolean, optional, default: False
        When True, delta loss will be checked after each iteration to stop learning earlier.

    num_threads: int, optional, default: 0
        Number of parallel threads for training. If num_threads=0, all CPU cores will be utilized.
        If seed is not None, num_threads=1 to remove randomness from parallelization. 
        (Only effective if using CPU backend).

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

    References
    ----------
    * Koren, Y., Bell, R., & Volinsky, C. Matrix factorization techniques for recommender systems. \
    In Computer, (8), 30-37. 2009.
    """

    def __init__(
        self,
        name="MF",
        k=10,
        backend="cpu",
        optimizer="sgd",
        max_iter=20,
        learning_rate=0.01,
        batch_size=256,
        lambda_reg=0.02,
        dropout=0.0,
        use_bias=True,
        early_stop=False,
        num_threads=0,
        trainable=True,
        verbose=False,
        init_params=None,
        seed=None,
    ):
        super().__init__(name=name, trainable=trainable, verbose=verbose)
        self.k = k
        self.backend = backend
        self.optimizer = optimizer
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.lambda_reg = lambda_reg
        self.dropout = dropout
        self.use_bias = use_bias
        self.early_stop = early_stop
        self.seed = seed

        if seed is not None:
            self.num_threads = 1
        elif num_threads > 0 and num_threads < multiprocessing.cpu_count():
            self.num_threads = num_threads
        else:
            self.num_threads = multiprocessing.cpu_count()

        # Init params if provided
        self.init_params = {} if init_params is None else init_params
        self.u_factors = self.init_params.get("U", None)
        self.i_factors = self.init_params.get("V", None)
        self.u_biases = self.init_params.get("Bu", None)
        self.i_biases = self.init_params.get("Bi", None)

    def _init(self):
        rng = get_rng(self.seed)

        if self.u_factors is None:
            self.u_factors = normal(
                [self.num_users, self.k], std=0.01, random_state=rng
            )
        if self.i_factors is None:
            self.i_factors = normal(
                [self.num_items, self.k], std=0.01, random_state=rng
            )

        self.u_biases = (
            zeros(self.num_users) if self.u_biases is None else self.u_biases
        )
        self.i_biases = (
            zeros(self.num_items) if self.i_biases is None else self.i_biases
        )
        self.global_mean = self.global_mean if self.use_bias else 0.0

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
            if self.backend == "cpu":
                self._fit_cpu(train_set, val_set)
            elif self.backend == "pytorch":
                self._fit_pt(train_set, val_set)
            else:
                raise ValueError(f"{self.backend} is not supported")
        return self

    #################
    ## CPU backend ##
    #################
    def _fit_cpu(self, train_set, val_set):
        from cornac.models.mf import backend_cpu

        (rid, cid, val) = train_set.uir_tuple
        backend_cpu.fit_sgd(
            rid,
            cid,
            val.astype(np.float32),
            self.u_factors,
            self.i_factors,
            self.u_biases,
            self.i_biases,
            self.num_users,
            self.num_items,
            self.learning_rate,
            self.lambda_reg,
            self.global_mean,
            self.max_iter,
            self.num_threads,
            self.use_bias,
            self.early_stop,
            self.verbose,
        )

    #####################
    ## PyTorch backend ##
    #####################
    def _fit_pt(self, train_set, val_set):
        import torch
        from .backend_pt import MF, learn

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        if self.seed is not None:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)

        model = MF(
            self.u_factors,
            self.i_factors,
            self.u_biases.reshape(-1, 1),
            self.i_biases.reshape(-1, 1),
            self.use_bias,
            self.global_mean,
            self.dropout,
        )

        learn(
            model=model,
            train_set=train_set,
            n_epochs=self.max_iter,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            reg=self.lambda_reg,
            optimizer=self.optimizer,
            device=device,
        )

        self.u_factors = model.u_factors.weight.detach().cpu().numpy()
        self.i_factors = model.i_factors.weight.detach().cpu().numpy()
        if self.use_bias:
            self.u_biases = model.u_biases.weight.detach().cpu().squeeze().numpy()
            self.i_biases = model.i_biases.weight.detach().cpu().squeeze().numpy()

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
        if item_idx is not None and self.is_unknown_item(item_idx):
            raise ScoreException("Can't make score prediction for item %d" % item_idx)

        if item_idx is None:
            known_item_scores = self.global_mean + self.i_biases
            if self.knows_user(user_idx):
                known_item_scores += self.u_biases[user_idx]
                fast_dot(self.u_factors[user_idx], self.i_factors, known_item_scores)
            return known_item_scores
        else:
            item_score = self.global_mean + self.i_biases[item_idx]
            if self.knows_user(user_idx):
                item_score += self.u_biases[user_idx]
                item_score += self.u_factors[user_idx].dot(self.i_factors[item_idx])
            return item_score

    def get_vector_measure(self):
        """Getting a valid choice of vector measurement in ANNMixin._measures.

        Returns
        -------
        measure: MEASURE_DOT
            Dot product aka. inner product
        """
        return MEASURE_DOT

    def get_user_vectors(self):
        """Getting a matrix of user vectors serving as query for ANN search.

        Returns
        -------
        out: numpy.array
            Matrix of user vectors for all users available in the model.
        """
        user_vectors = self.u_factors
        if self.use_bias:
            user_vectors = np.concatenate(
                (user_vectors, np.ones([user_vectors.shape[0], 1])), axis=1
            )
        return user_vectors

    def get_item_vectors(self):
        """Getting a matrix of item vectors used for building the index for ANN search.

        Returns
        -------
        out: numpy.array
            Matrix of item vectors for all items available in the model.
        """
        item_vectors = self.i_factors
        if self.use_bias:
            item_vectors = np.concatenate(
                (item_vectors, self.i_biases.reshape((-1, 1))), axis=1
            )
        return item_vectors

    def recommend(self, user_ids, n=10, filter_history=True):
        """
        Provide recommendations for a list of users
        user_ids: list of users
        n: number of recommendations
        filter history: do not recommend items from users history
        Return: dataframe of users, items with the top n predictions
        """
        recommendation = []
        uir_df = pd.DataFrame(np.array(self.train_set.uir_tuple).T, columns=['user', 'item', 'rating'])
        uir_df['user'] = uir_df['user'].astype(int)
        uir_df['item'] = uir_df['item'].astype(int)
        item_idx2id= {v:k for k,v in self.train_set.iid_map.items()}

        for uid in user_ids:            
            if uid not in self.train_set.uid_map:
                continue
            user_idx = self.train_set.uid_map[uid]
            item_rank, item_score = self.rank(user_idx)
            recommendation_one_user = []
            if filter_history:
                user_rated_items = uir_df.loc[uir_df['user'] == user_idx]['item']
                # remove user rated items from item_rank
                recommendation_one_user = [[uid, item_idx2id[item_idx], item_score[item_idx]] for item_idx in item_rank if item_idx not in user_rated_items][:n]
            else:
                recommendation_one_user = [[uid, item_idx2id[item_idx], item_score[item_idx]] for item_idx in item_rank[:n]]
            recommendation.extend(recommendation_one_user)
        
        return pd.DataFrame(recommendation, columns=['user_id', 'item_id', 'prediction'])