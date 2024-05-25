import numpy as np
import pandas as pd
import torch
from torch.optim import SGD
from scipy.optimize import minimize
from .explainer import Explainer
from tqdm.auto import tqdm
from scipy.sparse import csr_matrix


class Exp_Counter(Explainer):
    """Explainer for Explainable Recommendation with Comparative Constraints on Subjective Aspect-Level Quality

    Parameters
    ----------
    rec_model: object, recommender model
        The recommender model to be explained.
    dataset: object, dataset
        The dataset object that is used to explain.
    alpha: float, optional, default: 0.3
        The weight of the hinge loss.
    gamma: float, optional, default: 1.0
        The weight of the L1 regularization.
    lr: float, optional, default: 0.01
        The learning rate for optimization.
    max_iter: int, optional, default: 10
        The maximum number of iterations for optimization.
    rec_k: int, optional, default: 10
        The k+1th recommended item's score will be used as the benchmark score.
    name: string, optional, default: 'Exp_ComparERSub'

    References
    ----------
    [1] Tan, Juntao and Xu, Shuyuan and Ge, Yingqiang and Li, Yunqi and Chen, Xu and Zhang, Yongfeng. 2021. Counterfactual Explainable Recommendation
    Proceedings of the 30th ACM International Conference on Information & Knowledge Management

    [2] https://github.com/chrisjtan/counter/tree/main
    """

    def __init__(self, rec_model, dataset, alpha=0.3, gamma=1.0, lamda=100, lr=0.01, rec_k=5, max_iter=20, name="Exp_Counter", verbose=False):
        super().__init__(name, rec_model, dataset)
        self.alpha = alpha
        self.gamma = gamma
        self.lamda = lamda
        self.lr = lr
        self.max_iter = max_iter
        self.rec_k = rec_k
        self.verbose = verbose

        if self.model is None:
            raise ValueError("The model cannot be None.")
        
    def _train(self):
        counter = CounterTrainer(self.model, self.dataset, self.user_id, self.item_id, self.alpha, self.gamma, self.rec_k)
        optimizer = SGD(counter.parameters(), lr=self.lr)
        counter.train()
        best_delta = counter.delta.detach().numpy().flatten()
        min_loss = float("inf")
        is_valid_delta = False
        for epoch in range(self.max_iter):
            optimizer.zero_grad()
            score = counter()
            is_valid, loss = counter.loss(score)
            loss.backward()
            optimizer.step()
            if loss.item() < min_loss:
                min_loss = loss.item()
                best_delta = counter.delta.detach().numpy().flatten()
                is_valid_delta = is_valid
            
            if self.verbose:
                print(f"Epoch: {epoch}, Loss: {loss.item()}")
            # earlystop when find valid delta
            if is_valid_delta:
                break

        return is_valid_delta, best_delta

    def explain_one_recommendation_to_user(self, user_id, item_id, **kwargs):
        
        self.feature_k = kwargs.get("feature_k", 3)
        self.user_id = self.dataset.uid_map[user_id]
        self.item_id = self.dataset.iid_map[item_id]
        is_valid_delta, best_delta = self._train()
        explanation = {}
        
        top_k_features_ids = np.argsort(best_delta)[:self.feature_k]
        id_aspect_map = {v: k for k, v in self.dataset.sentiment.aspect_id_map.items()}
        
        for i in top_k_features_ids:
            feature_name = id_aspect_map[i]
            explanation[feature_name] = best_delta[i]
    
        return explanation
        
        

class CounterTrainer(torch.nn.Module):
    def __init__(self, rec_model, dataset, user_id, item_id, alpha=0.2, gamma=1.0, lamda=50, rec_k=10, verbose=False):
        super(CounterTrainer, self).__init__()
        self.rec_model = rec_model
        self.dataset = dataset
        self.alpha = alpha
        self.gamma = gamma
        self.lamda = lamda
        self.user_id = user_id
        self.item_id = item_id
        self.rec_k = rec_k
        if self.rec_model.name == "EFM":
            _, _, self.initial_Y = self.rec_model._build_matrices(self.dataset)
        elif self.rec_model.name == "MTER":
            (_, _, _, _, item_aspect_opinion) = self.rec_model._build_data(self.dataset)
            raise NotImplementedError(f"{self.name} for {self.rec_model.name} has not been implemented yet.")
        init_delta = np.random.uniform(-1, 0, self.initial_Y.shape[1]).reshape(1, -1)
        # self.delta = torch.nn.Parameter(torch.FloatTensor(np.random.uniform(-1, 0, self.initial_Y.shape[1]).reshape(1, -1)))
        # mask the delta to zero if the data in Y is zero
        init_delta[self.initial_Y[self.item_id].toarray() == 0] = 0
        self.delta = torch.nn.Parameter(torch.FloatTensor(init_delta))
        
        self._set_benchmark_score()
        
    def _set_benchmark_score(self):
        ranked_items, item_scores = self.rec_model.rank(self.user_id)
        self.s_ij_Kplus1 = item_scores[ranked_items[self.rec_k + 1]]
    
    def forward(self):
        # self._refit_model()
        Y = self.initial_Y.copy()
        Y[self.item_id] += self.delta.detach().numpy().reshape((1, -1))
        self.rec_model.fit(self.dataset, delta_Y=Y)
        score = self.rec_model.score(self.user_id, self.item_id)
        
        return score
    
    def loss(self, score):
        l2 = torch.linalg.norm(self.delta, ord=2)
        l1 = torch.linalg.norm(self.delta, ord=1)
        s_ij_delta = score
        
        hinge_loss = torch.clamp(torch.tensor(self.alpha) + torch.tensor(s_ij_delta) - torch.tensor(self.s_ij_Kplus1), min=0)
        loss = l2 + l1 * self.gamma + hinge_loss*self.lamda
        is_valid_delta = True if s_ij_delta < self.s_ij_Kplus1 else False
        
        return is_valid_delta, loss
    
    # def _refit_model(self):
    #     if self.rec_model.name == "EFM":
    #         Y = self.initial_Y.copy
    #         Y[self.item_id] += self.delta
    #         A_user_counts = np.ediff1d(self.A.indptr)
    #         A_item_counts = np.ediff1d(self.A.tocsc().indptr)
    #         A_uids = np.repeat(np.arange(self.dataset.num_users), A_user_counts).astype(self.A.indices.dtype)
    #         X_user_counts = np.ediff1d(self.X.indptr)
    #         X_aspect_counts = np.ediff1d(self.X.tocsc().indptr)
    #         X_uids = np.repeat(np.arange(self.dataset.num_users), X_user_counts).astype(self.X.indices.dtype)
    #         Y_item_counts = np.ediff1d(Y.indptr)
    #         Y_aspect_counts = np.ediff1d(Y.tocsc().indptr)
    #         Y_iids = np.repeat(np.arange(self.dataset.num_items), Y_item_counts).astype(Y.indices.dtype)
            
    #         self.rec_model._fit_efm(
    #             self.rec_model.num_threads,
    #             self.A.data.astype(np.float32), A_uids, self.A.indices, A_user_counts, A_item_counts,
    #             self.X.data.astype(np.float32), X_uids, self.X.indices, X_user_counts, X_aspect_counts,
    #             Y.data.astype(np.float32), Y_iids, Y.indices, Y_item_counts, Y_aspect_counts,
    #             self.rec_model.U1, self.rec_model.U2, self.rec_model.V, self.rec_model.H1, self.rec_model.H2
    #         )
    #         return self.rec_model
    #     elif self.rec_model.name == "MTER":
    #         raise NotImplementedError("MTER is not supported.")
    #     else:
    #         raise NotImplementedError(f"{self.rec_model.name} Model is not supported.")
       