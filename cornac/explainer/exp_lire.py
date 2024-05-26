import torch
import scipy
# import umap
import umap.umap_ as umap

import numpy as np

from tqdm import tqdm
from torch import optim, nn
from sklearn import linear_model
from sklearn.cluster import KMeans
from .explainer import Explainer

class Exp_LIRE(Explainer):
    """
    LIRE: Local and Interpretable Recommendations Explanations
    
    Parameters
    ----------
    rec_model: object
        A recommender model object that has been trained on the dataset.
    dataset: object
        A dataset object that has been used to train the recommender model.
    name: str, optional, default: "Exp_LIRE"
        Name of the explainer.
    verbose: bool, optional, default: False
        If True, print running logs.

    References
    ----------
    Brunot, L., Canovas, N., Chanson, A., Labroche, N., & Verdeaux, W. (2022).
    Preference-based and local post-hoc explanations for recommender systems.
    Information Systems, 108, 102021.
    """

    def __init__(
            self,
            rec_model, 
            dataset,
            name="Exp_LIRE",
            verbose=False
    ):
        super().__init__(name=name, rec_model=rec_model, dataset=dataset)
        self.u_factors = None
        self.i_factors_T = None
        self.sigma = None
        self.user_means = None
        self.global_variance = None
        self.user_item_matrix = None

        self.setup = False
        self.verbose = verbose


    def _set_up_explainer(
            self,
            n_clusters=75,
            n_components=3,
            n_neighbors=15,
            min_dist=0.1,
        ):
        """
        Set up the explainer by first fetching the user and item factors from the rec model.
        Then create clusters of users based on their latent factors.
        Employes UMAP for dimensionality reduction and KMeans for clustering.

        Args:
            n_clusters: number of clusters to be created
            n_components: number of components to be considered in UMAP
            n_neighbors: number of neighbors to be considered in UMAP
            min_dist: minimum distance between points in UMAP
        """

        self.u_factors = self.model.u_factors
        self.i_factors_T = self.model.i_factors.T
        self.user_item_matrix = self.model.train_set.matrix
        self.global_variance = self.user_item_matrix.data.var()
        self.user_means = _get_user_means(self.user_item_matrix)

        if self.verbose:
            print("Running UMAP")
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            low_memory=True
        )
        embedding = reducer.fit_transform(self.user_item_matrix)

        if self.verbose:
            print("Running KMeans")

        kmeans = KMeans(n_clusters=n_clusters).fit(embedding)
        self.cluster_labels = kmeans.labels_

        if self.verbose:
            print("Setup complete")
        self.setup = True

    def explain_one_recommendation_to_user(
            self,
            user_id,
            item_id,
            **kwargs
    ):
        """
        Provide explanations for one instance of (user_id, item_id), explain by top features and their coefficients.

        Parameters
        ----------
        user_id: str
            One user's id.
        item_id: str
            One item's id.
        feature_k: int, optional, default:10
            Number of features in explanations created by explainer.
        pert_ratio: float, optional, default: 0.5
            Ratio of perturbed users to real neighbors in the training set.
        train_set_size: int, optional, default: 1000
            Size of the training set for the local surrogate model.
    

        Returns
        -------
        explanation: dict
            Explanations as a dictionary as {item_id: coefficient}

        """
        if not self.setup:
            self._set_up_explainer()
            
        num_features = kwargs.get("feature_k", 10)
        pert_ratio = kwargs.get("pert_ratio", 0.5)
        train_set_size = kwargs.get("train_set_size", 1000)

        explanation = {}
        if user_id not in self.dataset.uid_map:
            print(f"User {user_id} not in dataset")
            return {}
        if item_id not in self.dataset.iid_map:
            print(f"Item {item_id} not in dataset")
            return {}
        # Map the user_id and item_id to the corresponding matrix ids
        user_matrix_idx = self.dataset.uid_map[user_id]
        item_matrix_idx = self.dataset.iid_map[item_id]
        if self.model.is_unknown_user(user_matrix_idx):
            return {}
        if self.model.is_unknown_item(item_matrix_idx):
            return {}

        # 1. Generate a train set for local surrogate model
        X_train = np.zeros((train_set_size, self.i_factors_T.shape[1]))
        y_train = np.zeros(train_set_size)

        # nb of perturbed entries
        pert_nb = int(train_set_size * pert_ratio)
        # nb of real neighbors
        cluster_nb = train_set_size - pert_nb

        # Neighbors by perturbation
        if pert_nb > 0:
            # generate perturbed users based on gaussian noise and store in X_train
            X_train[0:pert_nb, :] = _perturbations_gaussian(
                self.user_item_matrix[user_matrix_idx],
                self.global_variance,
                pert_nb
            )
            X_train[0:pert_nb, item_matrix_idx] = 0
            
            # isolate the perturbed users
            users = X_train[range(pert_nb)]
            # Make the predictions for those
            OOS_predictions = _OOS_pred_slice(
                users,
                self.u_factors,
                self.i_factors_T,
                self.user_means[user_matrix_idx],
            ).detach().numpy()[:, item_matrix_idx]
            # store the predictions in the train data
            y_train[range(pert_nb)] = OOS_predictions

        # Neighbors from the same cluster
        if cluster_nb > 0:
            # generate neighbors training set part
            cluster_index = self.cluster_labels[user_matrix_idx]
            # retrieve the cluster index of user "user_id"
            neighbors_index = np.where(self.cluster_labels == cluster_index)[0]
            # remove the user_id itself from the neighbors
            neighbors_index = neighbors_index[neighbors_index != user_matrix_idx]
            if(len(neighbors_index)>1):
                neighbors_index = np.random.choice(neighbors_index, cluster_nb)
                X_train[pert_nb:train_set_size, :] = self.user_item_matrix.toarray()[neighbors_index, :]
                X_train[pert_nb:train_set_size, item_matrix_idx] = 0

                neighbor_pred = []
                for neighbor in neighbors_index:
                    # Make the predictions for those
                    neighbor_pred.append(self.model.score(neighbor, item_matrix_idx))
                y_train[pert_nb:train_set_size] = neighbor_pred

        X_user_id = self.user_item_matrix.toarray()[user_matrix_idx].copy()
        X_user_id[item_matrix_idx] = 0

        # Check the real black box prediction for the user_id, item_id
        # real_pred = self.model.score(user_matrix_id, item_matrix_id)

        # Run a LARS linear regression on the train set to generate the most
        # parcimonious explanation
        reg = linear_model.Lars(fit_intercept=False, n_nonzero_coefs=num_features,eps=90)
        reg.fit(X_train, y_train)
        coef = reg.coef_
        # Predict the value with the surrogate
        # pred = reg.predict(X_user_id.reshape(1, -1))

        # movie ids in the dataset
        # movielens_ids = list(self.dataset.iid_map.keys())
        # mapped movie ids in the matrix
        # matrix_ids = list(self.dataset.iid_map.values())

        # get coefficients and corresponding movie ids
        top_features_matrix_ids = np.argsort(np.abs(coef))[::-1][:num_features]
        # top_features_movielens_ids = [
        #     movielens_ids[matrix_ids.index(i)] for i in top_features_matrix_ids
        # ]
        # top_features_coef = coef[top_features_matrix_ids]

        for i in top_features_matrix_ids:
            feature_id = self.dataset.item_ids[i]
            explanation[feature_id] = coef[i]
        
        return explanation
        

"""
LIRE utils, not part of the class.
"""

def _perturbations_gaussian(original_user, global_variance, fake_users: int, proba=0.1):
    """
    Function that does the gaussian perturbation and therefore yield perturbated points 
    that is supposedly close to the instance to explain
    """
    if isinstance(original_user, scipy.sparse.csr_matrix):
        original_user = original_user.toarray()
    else:
        original_user = original_user.reshape(1, len(original_user))
    # Comes from a scipy sparse matrix
    nb_dim = original_user.shape[1]
    users = np.tile(original_user, (fake_users, 1))

    noise = np.random.normal(np.zeros(nb_dim), global_variance/2, (fake_users, nb_dim))
    rd_mask = np.random.binomial(1, proba, (fake_users, nb_dim))
    noise = noise * rd_mask * (users != 0.)
    users = users + noise
    return np.clip(users, 0., 5.)


def _OOS_pred_slice(users, u_factors, i_factors_T, user_means, epochs=120, init_vec=None):
    """
    Function that does the out of sample prediction for a slice of the user matrix

    Args:
        users: the slice of the user matrix
        u_factors: the user factors
        i_factors_T: the item factors transposed (k x n_items)
        user_means: the mean ratings of the users
        epochs: the number of epochs for the training
        init_vec: the initial vector for the user factors
    """
    # Convert to torch tensors
    users = torch.tensor(users, dtype=torch.float32)
    u_factors = torch.tensor(u_factors, dtype=torch.float32)
    i_factors_T = torch.tensor(i_factors_T, dtype=torch.float32)    
    user_means = torch.tensor(user_means, dtype=torch.float32)

    if init_vec:
        unew = nn.Parameter(
            torch.tensor(
                init_vec,
                device=users.device,
                dtype=users.dtype,
                requires_grad=True
            )
        )
    else:
        unew = nn.Parameter(
            torch.ones(
                users.size()[0],
                u_factors.size()[0],
                device=users.device, dtype=users.dtype, requires_grad=True
            )
        )

    opt = optim.Adagrad([unew], lr=0.1)

    for epoch in range(epochs):
        pred = torch.matmul(torch.matmul(unew, u_factors), i_factors_T) + user_means.unsqueeze(1) * (users == 0.) + users
        loss = torch.sum(torch.pow((users - pred), 2)) / torch.numel(users)
        loss.backward()

        opt.step()
        opt.zero_grad()

    return ((torch.matmul(torch.matmul(unew, u_factors), i_factors_T) + user_means.unsqueeze(1)) * (users == 0.) + users).detach().clamp(0., 5.)


def _get_user_means(user_item_matrix):
    """
    Function that computes the mean of the users
    """
    user_means = [None] * user_item_matrix.shape[0]
    for line, col in user_item_matrix.todok().keys():
        if user_means[line] is None:
            user = user_item_matrix[line].toarray()
            user[user == 0.] = np.nan
            user_means[line] = np.nanmean(user)
    user_means = np.array(user_means)
    user_means = user_means.reshape(user_item_matrix.shape[0],1)
    return user_means