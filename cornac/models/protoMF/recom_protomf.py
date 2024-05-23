from ..recommender import Recommender
from .feature_extractors import FeatureExtractorFactory
from .protoMF import ProtoMFModel
from .dataset import ProtoRecDataset
from ...exception import ScoreException
from .standard_params import Std_params
from .evaluator import Evaluator
from .explainer_functions import get_top_k_items


import torch
import torch.nn as nn
from torch.utils import data
import numpy as np
from tqdm import tqdm
import time
import pandas as pd
from functools import partial

class ProtoMF(Recommender):
    """
    Proto MF
    This class implements the ProtoMF model as proposed in the below mentioned paper. ProtoMF uses prototypes to represent
    users or items. These prototypes help to increase the recommendations reasoning as they are interpretable for everyone. 
    The procedure of predictions consists of different main parts, namely, the creation of prototypes which can subsequently
    be used for user/item embedding and  the computation part in which the model gets trained on training data.

    Parameters
    ----------
    ft_ext_param: dictionary
        Dictionary holding all hyperparameters for the appropriate model, as hyperparameter tuning is currently omitted
        due to low computational resources, the dictionaries can be manually adjusted in the standard_params file

    n_epochs: int, optional, default: according to predefined standard parameters in the standard_params file
        Maximum number of epochs

    learning_rate: float, optional, default: according to predefined standard parameters in the standard_params file
        The learning rate

    weight_decay: float, optional, default: according to predefined standard parameters in the standard_params file
        The weight decay

    optim_name: string, optional, default: according to predefined standard parameters in the standard_params file
        The name of the optimizer used to train the model

    loss: string, optional, default: according to predefined standard parameters in the standard_params file
        The name of the loss on which we optimize the model

    max_patience: int, optional, default: according to predefined standard parameters in the standard_params file
        The number of epochs that are accepted without an improvement before the model stops


    References
    ----------
    * Melchiorre, A.B., Rekabsaz, N., Ganhör, C. and Schedl, M.,
      2022, September. Protomf: Prototype-based matrix factorization for effective and explainable
      recommendations. In Proceedings of the 16th ACM Conference on Recommender Systems (pp. 246-256).
    """

    def __init__(self,
                 ft_ext_param,
                 n_epochs=Std_params.N_EPOCHS,
                 learning_rate=Std_params.LEARNING_RATE,
                 weight_decay=Std_params.WEIGHT_DECAY,
                 optim_name=Std_params.OPTIM_NAME,
                 loss=Std_params.LOSS,
                 max_patience=Std_params.MAX_PATIENCE,
                 ):
        
        Recommender.__init__(self, name=ft_ext_param["name"])
        self.ft_ext_param = ft_ext_param
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_patience = max_patience
        self.n_epochs = n_epochs
        self.optim_name = optim_name


        if loss == 'bce':
            print("Using loss function: Binary cross entropy")
            self.loss_function = partial(bce_loss)
        elif loss == 'bpr':
            print("Using loss function: Bayesian Personalized Ranking")
            self.loss_function = partial(bpr_loss) 
        elif loss == 'sampled_softmax':
            print("Using loss function: (Sampled) Softmax")
            self.loss_function = partial(sampled_softmax_loss)
        else:
            raise ValueError(f'Recommender System Loss function <{loss}> Not Implemented')

    def _build_model(self, n_users, n_items):
        """This function builds a model for ProtoMF or any other specified model (MF etc.)

        Parameters
        ----------
        n_users: int
            Number of users

        n_items: int
            Number of items

        Returns
        -------
        protoMF: ProtoMFModel
            An object of our ProtoMF model
        """

        # Step 1 --- Building User and Item Feature Extractors (user/item embedding)
        """
        This is the step in which we differentiate between the different types of models, depending on the
        parameters in the ft_ext_param dictionary, user and item will either be embedded with traditional
        embedding or by transforming them into similarity vectors to the prototypes
        """
        user_feature_extractor, item_feature_extractor = FeatureExtractorFactory.create_models(self.ft_ext_param, n_users, n_items)

        # Step 2 --- Building RecSys Module
        protoMF = ProtoMFModel(n_users, n_items, user_feature_extractor, item_feature_extractor)


    def fit(self, train_set, val_set=None):
        """
        Fits the model to observations.

        Parameters
        ----------
        train_set: :obj:`cornac.data.MultimodalTrainSet`, required
            User-Item preference data as well as additional modalities.

        val_set: :obj:`cornac.data.MultimodalTestSet`, optional, default: None
            User-Item preference data for model selection purposes (e.g., early stopping).

        Returns
        -------
        self : object
        """
        self.train_set = train_set
        self.val_set = val_set

        # Create a model
        n_users, n_items = train_set.num_users, train_set.num_items
        self.model = self._build_model(n_users, n_items)

        # Initialize the optimizer
        if self.optim_name == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optim_name == 'adagrad':
            self.optimizer = torch.optim.Adagrad(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        else:
            raise ValueError('Optimizer not available')

        print(f'Built Optimizer  \n'
              f'- name: {self.optim_name} \n'
              f'- lr: {self.learning_rate} \n'
              f'- wd: {self.weight_decay} \n')

        # Create data loaders for data
        protorec_dataset_train = ProtoRecDataset(train_set, 'train', n_neg=10)
        self.train_loader = data.DataLoader(protorec_dataset_train, batch_size=256, shuffle=True)

        if val_set:
            protorec_dataset_val = ProtoRecDataset(val_set, 'val', n_neg=10)
            self.val_loader = data.DataLoader(protorec_dataset_val, batch_size=256, shuffle=True)
        else:
            self.val_loader = None

        # Always store the metrics for the best model so far, at the beginning use the default model 
        metrics_values = self.compute_metrics()
        # Print the current values of all metrics
        print(metrics_values)

        # Store the best value to check for improvements later
        # After computing the metrics, choose the metric, you want to use for Early Stopping
        best_value = metrics_values["hit_ratio@10"]
        print('Init - Avg Val Value {:.3f} \n'.format(best_value))

        patience = 0
        
        # Iterate over all epochs and evaluate the model on the validation data after each epoch
        # Always store the parameter of the best model to be able to reconstruct it afterwards
        for epoch in tqdm(range(self.n_epochs)):

            # For early stopping, we evaluate on the validation data and stop if we haven't found a better model after x epochs
            if patience == self.max_patience:
                print('Max Patience reached, stopping.')
                break

            self.model.train()
            epoch_loss = 0.0

            # get a batch if indices to use with the yield statement
            start_time = time.time()
            n_batches = 0

            for u_idxs, i_idxs, labels in self.train_loader:

                n_batches += 1

                # Forward pass
                out = self.model(u_idxs, i_idxs)
                
                # Compute the loss and add it this epoch's loss
                loss = self.loss_function(out, labels)
                epoch_loss += loss.item()

                # Backward pass
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            end_time = time.time()
            print(f"Epoch {epoch+1}/{self.n_epochs}")
            print(f"Duration: {end_time - start_time} seconds")
            print(f"Loss: {epoch_loss/n_batches}")

            # Compute the metrics for this model
            metrics_values = self.compute_metrics()
            current_value = metrics_values['hit_ratio@10']


            # Finally, if the current loss is lower than the best so far, set a new one, else increase the patience by 1
            if current_value > best_value:
                best_value = current_value
                print('Epoch {} - New best model found (val value {:.3f}) \n'.format(epoch+1, current_value))

                patience = 0
            else:
                patience += 1
        # Finally, compute the item_weight matrices for explanations
        """if "User" in  self.ft_ext_param["name"]:
            self.item_weights_protoU = self.model.item_feature_extractor
        if "Item" in self.ft_ext_param["name"]:
            self.item_weights_protoI = ..."""


    def score(self, user_idx, item_idx=None):
        """
        Predict the scores/ratings of one user for an item.

        Parameters
        ----------
        user_id: int, required
            The index of the user for whom to perform score prediction.

        item_id: int, optional, default: None
            The index of the item for that to perform score prediction.
            If None, scores for all known items will be returned.

        Returns
        -------
        pred : A scalar or a Numpy array
            Relative scores that the user gives to the item or to all known items
        """

        self.model.eval()

        if not torch.is_tensor(user_idx):
            user_idx = torch.from_numpy(np.array([user_idx]))

        # Case 1: item_idx is None -> compute prediction for all known items
        if item_idx is None:
            if self.train_set.is_unk_user(user_idx):
                raise ScoreException(
                    "Can't make score prediction for (user_id=%d)" % user_idx
                )
            item_idxs = torch.tensor(list(self.train_set.item_data.keys()))
            pred = self.model(user_idx, item_idxs).detach().numpy()[0]
            return pred

        else:
            if self.train_set.is_unk_user(user_idx) or self.train_set.is_unk_item(item_idx):
                raise ScoreException(
                    "Can't make score prediction for (user_id=%d, item_id=%d)"
                    % (user_idx, item_idx)
                )

            if not torch.is_tensor(item_idx):
                item_idx = torch.from_numpy(np.array([item_idx]))

            user_pred = self.model(user_idx, item_idx).item()
            return user_pred


    # Ensure that no gradient is computed, saves memory
    @torch.no_grad()
    def compute_metrics(self):
        """
        Runs the evaluation procedure.

        Parameters
        ----------

        Returns
        -------        
        metrics_values: float
            output of the validation (e.g. NDCG@10).
        """
        # Set model to evaluation mode
        self.model.eval()

        print('Validation started')

        # Initialize the sum of losses as 0 (then add each loss, ultimately it will be divided by the count)
        val_loss = 0

        # Creates an evaluator object with the number of users in the validation dataset
        eval = Evaluator(self.val_loader.dataset.n_users)

        # Go over all batches and compute the loss
        for u_idxs, i_idxs, labels in self.val_loader:

            # Makes a forward pass with the given ID's meaning, it computes the prediction, returns a float
            out = self.model(u_idxs, i_idxs)

            # Add the loss computed with the desired loss function to our val_loss
            val_loss += self.loss_function(out, labels).item()

            # Apply a sigmoid function in order to squeeze the out values into the span [0, 1]
            out = nn.Sigmoid()(out)

            eval.eval_batch(out)

        # Divide by the number of pairings we have in the validation data
        val_loss /= len(self.val_loader)
        # Store the metric-values and the loss
        metrics_values = {**eval.get_results(), 'val_loss': val_loss}

        return metrics_values

        
    def recommend(self, user_ids, n=10, filter_history=True):
        """
        Provide recommendations for a list of users
                
        Parameters
        ----------
        user_ids: list
            A list of users
        n: int
            Number of recommendations
        filter history: boolean
            If True: do not recommend items from users history
         
        Returns
        ------- 
        A dataframe of users, items with the top n predictions
        """
        # Access to the embedding layer after model has been trained, can be used for user prototype interpretation
        # self.model.item_feature_extractor.embedding_layer.weight
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
                user_rated_items = uir_df[uir_df['user'] == user_idx]['item']
                # remove user rated items from item_rank
                recommendation_one_user = [[uid, item_idx2id[item_idx], item_score[item_idx]] for item_idx in item_rank if item_idx not in user_rated_items][:n]
            else:
                recommendation_one_user = [[uid, item_idx2id[item_idx], item_score[item_idx]] for item_idx in item_rank[:n]]
            recommendation.extend(recommendation_one_user)
        
        return pd.DataFrame(recommendation, columns=['user_id', 'item_id', 'prediction'])




def bce_loss(logits, labels, aggregator='mean'):
    """
    It computes the binary cross entropy loss with negative sampling, expressed by the formula:
                                    -∑_j log(x_ui) + log(1 - x_uj)
    where x_ui and x_uj are the prediction for user u on item i and j, respectively. Item i positive instance while
    Item j is a negative instance. The Sum is carried out across the different negative instances. In other words
    the positive item is weighted as many as negative items are considered.
                
    Parameters
    ----------
    logits: array
        An array of logits values from the network. The first column always contain the values of positive instances.
        Shape is (batch_size, 1 + n_neg).
    labels: array
        1-0 Labels. The first column contains 1s while all the others 0s.
    aggregator: String, optional, default: "mean"
        function to use to aggregate the loss terms

    Returns
    ------- 
    The binary cross entropy as computed above
    """
    weights = torch.ones_like(logits)
    weights[:, 0] = logits.shape[1] - 1

    loss = nn.BCEWithLogitsLoss(weights.flatten(), reduction=aggregator)(logits.flatten(), labels.flatten())

    return loss

def bpr_loss(logits, labels, aggregator='mean'):
    """
    It computes the Bayesian Personalized Ranking loss (https://arxiv.org/pdf/1205.2618.pdf).

    Parameters
    ----------
    logits: array
        An array of logits values from the network. The first column always contain the values of positive instances.
        Shape is (batch_size, 1 + n_neg).
    labels: array
        1-0 Labels. The first column contains 1s while all the others 0s.
    aggregator: String, optional, default: "mean"
        function to use to aggregate the loss terms

    Returns
    ------- 
    The bayesian personalized ranking loss
    """
    pos_logits = logits[:, 0].unsqueeze(1)  # [batch_size,1]
    neg_logits = logits[:, 1:]  # [batch_size,n_neg]

    labels = labels[:, 0]
    labels = torch.repeat_interleave(labels, neg_logits.shape[1])

    diff_logits = pos_logits - neg_logits

    loss = nn.BCEWithLogitsLoss(reduction=aggregator)(diff_logits.flatten(), labels.flatten())

    return loss


def sampled_softmax_loss(logits, labels, aggregator='sum'):
    """
    It computes the (Sampled) Softmax Loss (a.k.a. sampled cross entropy) expressed by the formula:
                        -x_ui +  log( ∑_j e^{x_uj})
    where x_ui and x_uj are the prediction for user u on item i and j, respectively. Item i positive instance while j
    goes over all the sampled items (negatives + the positive).
    
    Parameters
    ----------
    logits: array
        An array of logits values from the network. The first column always contain the values of positive instances.
        Shape is (batch_size, 1 + n_neg).
    labels: array
        1-0 Labels. The first column contains 1s while all the others 0s.
    aggregator: String, optional, default: "mean"
        function to use to aggregate the loss terms

    Returns
    -------
    The sampled softmax loss as computed above
    """

    pos_logits_sum = - logits[:, 0]
    log_sum_exp_sum = torch.logsumexp(logits, dim=-1)

    sampled_loss = pos_logits_sum + log_sum_exp_sum

    if aggregator == 'sum':
        return sampled_loss.sum()
    elif aggregator == 'mean':
        return sampled_loss.mean()
    else:
        raise ValueError('Loss aggregator not defined')