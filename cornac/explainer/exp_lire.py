import torch
import scipy
import umap

import pandas as pd
import numpy as np

from tqdm import tqdm
from torch import optim, nn
from sklearn import linear_model
from sklearn.cluster import KMeans
from scipy.sparse import csr_matrix

from cornac.explainer import Explainer


class LIRE(Explainer):
    """
    LIRE: Local and Interpretable Recommendations Explanations

    Reference:
    ------------
    Brunot, L., Canovas, N., Chanson, A., Labroche, N., & Verdeaux, W. (2022).
    Preference-based and local post-hoc explanations for recommender systems.
    Information Systems, 108, 102021.
    """

    def __init__(
            self,
            rec_model, 
            dataset,
            name="LIRE",
            num_features=10,
            pert_ratio=0.5,
            verbose=False,
    ):
        self.rec_model=rec_model
        self.dataset=dataset
        self.name=name

        # paremeters for explanations
        self.pert_ratio = pert_ratio
        self.num_features = num_features

        self.u_factors = None
        self.i_factors_T = None
        self.sigma = None
        self.user_means = None
        self.global_variance = None
        self.user_item_matrix = None

        self.verbose = verbose

        self.setup = False


    def set_up_explainer(
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

        self.u_factors = self.rec_model.U_svd if self.rec_model.U_svd is not None else self.rec_model.u_factors
        self.i_factors_T = self.rec_model.Vt_svd if self.rec_model.Vt_svd is not None else self.rec_model.i_factors.T
        self.sigma = self.rec_model.Sigma_svd 
        self.user_means = self.rec_model.user_means
        self.user_item_matrix = self.rec_model.user_item_matrix
        self.global_variance = self.user_item_matrix.data.var()

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


    def explain_recommendations(
            self,
            recommendations,
            num_features,
            verbose=True,
    ):
        """
        Generate explanations for a list of recommendations
        Args:
            recommendations: df of [user_id, item_id]
            num_features: number of features used in the explanation

        Return: df of [user_id, item_id, explanations, local_prediction, black_box_prediction]
        """
        explanations = []
        self.recommendations = recommendations
        
        if not self.setup:
            self.set_up_explainer()

        total = recommendations.shape[0]

        with tqdm(total=total, disable=not verbose, desc="Computing explanations: ") as pbar:
            for i, row in self.recommendations.iterrows():
                explanations.append(
                    self.explain_one_recommendation_to_user(
                        row.user_id,
                        row.item_id,
                        num_features=num_features,
                        pert_ratio=self.pert_ratio,
                        train_set_size=1000,
                        verbose=self.verbose
                    )
                )
                pbar.update(1)
        # Concatenate the explanations        
        explanations = pd.concat(explanations, ignore_index=True)

        return explanations


    def explain_one_recommendation_to_user(
            self,
            user_id,
            item_id,
            num_features,
            pert_ratio,
            train_set_size=1000,
            verbose=True
    ):
        """
        Provide explanations for one instance of (user_id, item_id)

        Args:
            user_id: id of user to be explained
            item_id: id of item to be explained
            num_features: num of features to be included in the output

        Return: df of [user_id, item_id, explanations, local_prediction, black_box_prediction]
                with a single row

        """
        if not self.setup:
            self.set_up_explainer()

        # Map the user_id and item_id to the corresponding matrix ids
        user_matrix_id = self.dataset.uid_map[user_id]
        item_matrix_id = self.dataset.iid_map[item_id]

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
            X_train[0:pert_nb, :] = perturbations_gaussian(
                self.user_item_matrix[user_matrix_id],
                self.global_variance,
                pert_nb
            )
            X_train[0:pert_nb, item_id] = 0
            # isolate the perturbed users
            users = X_train[range(pert_nb)]
            # Make the predictions for those
            OOS_predictions = OOS_pred_slice(
                users,
                self.u_factors,
                self.i_factors_T,
                self.user_means[user_id],
            ).detach().numpy()[:, item_matrix_id]
            # store the predictions in the train data
            y_train[range(pert_nb)] = OOS_predictions

        # Neighbors from the same cluster
        if cluster_nb > 0:
            # generate neighbors training set part
            cluster_index = self.cluster_labels[user_matrix_id]
            # retrieve the cluster index of user "user_id"
            neighbors_index = np.where(self.cluster_labels == cluster_index)[0]
            # remove the user_id itself from the neighbors
            neighbors_index = neighbors_index[neighbors_index != user_matrix_id]
            if(len(neighbors_index)>1):
                neighbors_index = np.random.choice(neighbors_index, cluster_nb)
                X_train[pert_nb:train_set_size, :] = self.user_item_matrix.toarray()[neighbors_index, :]
                X_train[pert_nb:train_set_size, item_id] = 0

                neighbor_pred = []
                for neighbor in neighbors_index:
                    # Make the predictions for those
                    neighbor_pred.append(self.rec_model.score(neighbor, item_id))
                y_train[pert_nb:train_set_size] = neighbor_pred

        X_user_id = self.user_item_matrix.toarray()[user_matrix_id].copy()
        X_user_id[item_id] = 0

        # Check the real black box prediction for the user_id, item_id
        real_pred = self.rec_model.score(user_id, item_id)

        # Run a LARS linear regression on the train set to generate the most
        # parcimonious explanation
        reg = linear_model.Lars(fit_intercept=False, n_nonzero_coefs=num_features,eps=90)
        reg.fit(X_train, y_train)
        coef = reg.coef_
        # Predict the value with the surrogate
        pred = reg.predict(X_user_id.reshape(1, -1))

        # movie ids in the dataset
        movielens_ids = list(self.dataset.iid_map.keys())
        # mapped movie ids in the matrix
        matrix_ids = list(self.dataset.iid_map.values())

        # get coefficients and corresponding movie ids
        top_features_matrix_ids = np.argsort(np.abs(coef))[::-1][:num_features]
        top_features_movielens_ids = [
            movielens_ids[matrix_ids.index(i)] for i in top_features_matrix_ids
        ]
        top_features_coef = coef[top_features_matrix_ids]

        # build explanation dataframe
        return pd.DataFrame({
            'user_id': [user_id],
            'item_id': [item_id],
            'explanations': [list(zip(top_features_movielens_ids, top_features_coef))],
            'local_prediction': [pred[0]],
            'black_box_prediction': [real_pred]
        })

"""
LIRE utils, not part of the class.
"""

def perturbations_gaussian(original_user, global_variance, fake_users: int, proba=0.1):
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


def OOS_pred_slice(users, u_factors, i_factors_T, user_means, epochs=120, init_vec=None):
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
