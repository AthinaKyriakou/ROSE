from abc import abstractmethod, ABC
from typing import Tuple

import torch
import torch.nn as nn

def general_weight_init(m):
    """
    Weight initializor for the corresponding type of layer
    """
    if type(m) in [nn.Linear, nn.Conv2d]:
        if m.weight.requires_grad:
            torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if hasattr(m, 'bias') and m.bias is not None and m.bias.requires_grad:
                torch.nn.init.constant_(m.bias, 0)
    elif type(m) == nn.Embedding:
        if m.weight.requires_grad:
            torch.nn.init.normal_(m.weight)

    elif type(m) == nn.BatchNorm2d:
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)



#############################################################################################
# In the following, we define classes which build the actual feature extractors or embedding
# layer depending on the selected model
#############################################################################################

class FeatureExtractor(nn.Module, ABC):
    """
    Abstract class representing one of the possible FeatureExtractor models. See also FeatureExtractorFactory.
    """

    def __init__(self):
        super().__init__()
        self.cumulative_loss = 0.
        self.name = "FeatureExtractor"

    def init_parameters(self):
        """
        Initial the Feature Extractor parameters
        """
        pass

    def get_and_reset_loss(self) -> float:
        """
        Reset the loss of the feature extractor and returns the computed value
        :return: loss of the feature extractor
        """
        loss = self.cumulative_loss
        self.cumulative_loss = 0.
        return loss

    @abstractmethod
    def forward(self, o_idxs: torch.Tensor) -> torch.Tensor:
        """
        Performs the feature extraction process of the object.
        """
        pass


class Embedding(FeatureExtractor):
    """
    FeatureExtractor that represents an object (item/user) only given by its embedding.
    """

    def __init__(self, n_objects: int, embedding_dim: int, max_norm: float = None, only_positive: bool = False):
        """
        Standard Embedding Layer
        :param n_objects: number of objects in the system (users or items)
        :param embedding_dim: embedding dimension
        :param max_norm: max norm of the l2 norm of the embeddings.
        :param only_positive: whether the embeddings can be only positive
        """
        super().__init__()
        self.n_objects = n_objects
        self.embedding_dim = embedding_dim
        self.max_norm = max_norm
        self.only_positive = only_positive
        self.name = "Embedding"

        self.embedding_layer = nn.Embedding(self.n_objects, self.embedding_dim, max_norm=self.max_norm)
        print(f'Built Embedding model \n'
              f'- n_objects: {self.n_objects} \n'
              f'- embedding_dim: {self.embedding_dim} \n'
              f'- max_norm: {self.max_norm}\n'
              f'- only_positive: {self.only_positive}')

    def init_parameters(self):
        self.embedding_layer.apply(general_weight_init)

    def forward(self, o_idxs: torch.Tensor) -> torch.Tensor:
        # embedds an object (user/item)
        assert o_idxs is not None, f"Object Indexes not provided! ({self.name})"
        embeddings = self.embedding_layer(o_idxs)
        if self.only_positive:
            embeddings = torch.absolute(embeddings)
        return embeddings



class EmbeddingW(Embedding):
    """
    FeatureExtractor that places a linear projection after an embedding layer. Used for sharing weights.
    """

    def __init__(self, n_objects: int, embedding_dim: int, max_norm: float = None, out_dimension: int = None,
                 use_bias: bool = False):
        """
        :param n_objects: see Embedding
        :param embedding_dim: see Embedding
        :param max_norm: see Embedding
        :param out_dimension: Out dimension of the linear layer. If none, set to embedding_dim.
        :param use_bias: whether to use the bias in the linear layer.
        """
        super().__init__(n_objects, embedding_dim, max_norm)
        self.out_dimension = out_dimension
        self.use_bias = use_bias

        if self.out_dimension is None:
            self.out_dimension = embedding_dim

        self.name = 'EmbeddingW'
        self.linear_layer = nn.Linear(self.embedding_dim, self.out_dimension, bias=self.use_bias)

        print(f'Built Embeddingw model \n'
              f'- out_dimension: {self.out_dimension} \n'
              f'- use_bias: {self.use_bias} \n')

    def init_parameters(self):
        super().init_parameters()
        self.linear_layer.apply(general_weight_init)

    def forward(self, o_idxs: torch.Tensor) -> torch.Tensor:
        o_embed = super().forward(o_idxs)
        return self.linear_layer(o_embed)



class PrototypeEmbedding(FeatureExtractor):
    """
    ProtoMF building block. It represents an object (item/user) given the similarity with the prototypes.
    """

    def __init__(self, n_objects: int, embedding_dim: int, n_prototypes: int = None, use_weight_matrix: bool = False,
                 sim_proto_weight: float = 1., sim_batch_weight: float = 1.,
                 reg_proto_type: str = 'soft', reg_batch_type: str = 'soft', cosine_type: str = 'shifted',
                 max_norm: float = None):
        """
        :param n_objects: number of objects in the system (users or items)
        :param embedding_dim: embedding dimension
        :param n_prototypes: number of prototypes to consider. If none, is set to be embedding_dim.
        :param use_weight_matrix: Whether to use a linear layer after the prototype layer.
        :param sim_proto_weight: factor multiplied to the regularization loss for prototypes
        :param sim_batch_weight: factor multiplied to the regularization loss for batch
        :param reg_proto_type: type of regularization applied batch-prototype similarity matrix on the prototypes. Possible values are ['max','soft','incl']
        :param reg_batch_type: type of regularization applied batch-prototype similarity matrix on the batch. Possible values are ['max','soft']
        :param cosine_type: type of cosine similarity to apply. Possible values ['shifted','standard','shifted_and_div']
        :param max_norm: max norm of the l2 norm of the embeddings.

        """

        super(PrototypeEmbedding, self).__init__()

        self.n_objects = n_objects
        self.embedding_dim = embedding_dim
        self.n_prototypes = n_prototypes
        self.use_weight_matrix = use_weight_matrix
        self.sim_proto_weight = sim_proto_weight
        self.sim_batch_weight = sim_batch_weight
        self.reg_proto_type = reg_proto_type
        self.reg_batch_type = reg_batch_type
        self.cosine_type = cosine_type

        self.embedding_ext = Embedding(n_objects, embedding_dim, max_norm)

        if self.n_prototypes is None:
            self.prototypes = nn.Parameter(torch.randn([self.embedding_dim, self.embedding_dim]))
            self.n_prototypes = self.embedding_dim
        else:
            self.prototypes = nn.Parameter(torch.randn([self.n_prototypes, self.embedding_dim]))

        if self.use_weight_matrix:
            self.weight_matrix = nn.Linear(self.n_prototypes, self.embedding_dim, bias=False)

        # Cosine Type
        if self.cosine_type == 'standard':
            self.cosine_sim_func = nn.CosineSimilarity(dim=-1)
        elif self.cosine_type == 'shifted':
            self.cosine_sim_func = lambda x, y: (1 + nn.CosineSimilarity(dim=-1)(x, y))
        elif self.cosine_type == 'shifted_and_div':
            self.cosine_sim_func = lambda x, y: (1 + nn.CosineSimilarity(dim=-1)(x, y)) / 2
        else:
            raise ValueError(f'Cosine type {self.cosine_type} not implemented')

        # Regularization Batch
        if self.reg_batch_type == 'max':
            self.reg_batch_func = lambda x: - x.max(dim=1).values.mean()
        elif self.reg_batch_type == 'soft':
            self.reg_batch_func = lambda x: self._entropy_reg_loss(x, 1)
        else:
            raise ValueError(f'Regularization Type for Batch {self.reg_batch_func} not yet implemented')

        # Regularization Proto
        if self.reg_proto_type == 'max':
            self.reg_proto_func = lambda x: - x.max(dim=0).values.mean()
        elif self.reg_proto_type == 'soft':
            self.reg_proto_func = lambda x: self._entropy_reg_loss(x, 0)
        elif self.reg_proto_type == 'incl':
            self.reg_proto_func = lambda x: self._inclusiveness_constraint(x)
        else:
            raise ValueError(f'Regularization Type for Proto {self.reg_proto_type} not yet implemented')

        self._acc_r_proto = 0
        self._acc_r_batch = 0
        self.name = "PrototypeEmbedding"

        print(f'Built PrototypeEmbedding model \n'
              f'- n_prototypes: {self.n_prototypes} \n'
              f'- use_weight_matrix: {self.use_weight_matrix} \n'
              f'- sim_proto_weight: {self.sim_proto_weight} \n'
              f'- sim_batch_weight: {self.sim_batch_weight} \n'
              f'- reg_proto_type: {self.reg_proto_type} \n'
              f'- reg_batch_type: {self.reg_batch_type} \n'
              f'- cosine_type: {self.cosine_type} \n')

    @staticmethod
    def _entropy_reg_loss(sim_mtx, axis: int):
        o_coeff = nn.Softmax(dim=axis)(sim_mtx)
        entropy = - (o_coeff * torch.log(o_coeff)).sum(axis=axis).mean()
        return entropy

    @staticmethod
    def _inclusiveness_constraint(sim_mtx):
        '''
        NB. This method is applied only on a square matrix (batch_size,n_prototypes) and it return the negated
        inclusiveness constraints (its minimization brings more equal load sharing among the prototypes)
        '''
        o_coeff = nn.Softmax(dim=1)(sim_mtx)
        q_k = o_coeff.sum(axis=0).div(o_coeff.sum())  # [n_prototypes]
        entropy_q_k = - (q_k * torch.log(q_k)).sum()
        return - entropy_q_k

    def init_parameters(self):
        if self.use_weight_matrix:
            nn.init.xavier_normal_(self.weight_matrix.weight)

    def forward(self, o_idxs: torch.Tensor) -> torch.Tensor:
        """
        :param o_idxs: Shape is either [batch_size] or [batch_size,n_neg_p_1]
        :return:
        """
        assert o_idxs is not None, "Object indexes not provided"
        assert len(o_idxs.shape) == 2 or len(o_idxs.shape) == 1, \
            f'Object indexes have shape that does not match the network ({o_idxs.shape})'

        o_embed = self.embedding_ext(o_idxs)  # [..., embedding_dim]

        # Followingly, we compute the cosine similarity between an object and the prototype, to read more,
        # please refer to the following github repository or to my seminar paper
        # https://github.com/pytorch/pytorch/issues/48306
        sim_mtx = self.cosine_sim_func(o_embed.unsqueeze(-2), self.prototypes)  # [..., n_prototypes]

        if self.use_weight_matrix:
            w = self.weight_matrix(sim_mtx)  # [...,embedding_dim]
        else:
            w = sim_mtx  # [..., embedding_dim = n_prototypes]

        # Computing additional losses
        batch_proto = sim_mtx.reshape([-1, sim_mtx.shape[-1]])

        self._acc_r_batch += self.reg_batch_func(batch_proto)
        self._acc_r_proto += self.reg_proto_func(batch_proto)

        return w

    def get_and_reset_loss(self) -> float:
        acc_r_proto, acc_r_batch = self._acc_r_proto, self._acc_r_batch
        self._acc_r_proto = self._acc_r_batch = 0
        return self.sim_proto_weight * acc_r_proto + self.sim_batch_weight * acc_r_batch


class ConcatenateFeatureExtractors(FeatureExtractor):

    def __init__(self, model_1: FeatureExtractor, model_2: FeatureExtractor, invert: bool = False):
        super().__init__()

        """
        Concatenates the latent dimension (considered in position -1) of two Feature Extractors models.
        :param model_1: a FeatureExtractor model
        :param model_2: a FeatureExtractor model
        :param invert: whether to place the latent representation from the second model on top.
        """

        self.model_1 = model_1
        self.model_2 = model_2
        self.invert = invert

        self.name = 'ConcatenateFeatureExtractors'

        print(f'Built ConcatenateFeatureExtractors model \n'
              f'- model_1: {self.model_1.name} \n'
              f'- model_2: {self.model_2.name} \n'
              f'- invert: {self.invert} \n')

    def forward(self, o_idxs: torch.Tensor) -> torch.Tensor:
        o_repr_1 = self.model_1(o_idxs)
        o_repr_2 = self.model_2(o_idxs)

        if self.invert:
            return torch.cat([o_repr_2, o_repr_1], dim=-1)
        else:
            return torch.cat([o_repr_1, o_repr_2], dim=-1)

    def get_and_reset_loss(self) -> float:
        loss_1 = self.model_1.get_and_reset_loss()
        loss_2 = self.model_2.get_and_reset_loss()
        return loss_1 + loss_2

    def init_parameters(self):
        self.model_1.init_parameters()
        self.model_2.init_parameters()





###############################################################################################
# FEATURE EXTRACTOR FACTORIES
# Depending on the selected method, this class ensures the right feature extractors are created
###############################################################################################

class FeatureExtractorFactory:

    @staticmethod
    def create_models(ft_ext_param: dict, n_users: int, n_items: int) -> Tuple[FeatureExtractor, FeatureExtractor]:

        """
        Helper function to create both the user and item feature extractor. It either creates two detached
        FeatureExtractors or a single one shared by users and items.
        :param ft_ext_param: parameters for the user feature extractor model. ft_ext_param.ft_type is used for
            switching between models.
        :param n_users: number of users in the system.
        :param n_items: number of items in the system.
        :return: [user_feature_extractor, item_feature_extractor]
        """
        assert 'ft_type' in ft_ext_param, "Type has not been specified for FeatureExtractor! " \
                                          "FeatureExtractor model not created"
        ft_type = ft_ext_param['ft_type']
        embedding_dim = ft_ext_param['embedding_dim']

        # For matrix factorization
        if ft_type == 'detached':
            # Build the extractors independently (e.g. two embeddings branches, one for users and one for items)
            user_feature_extractor = FeatureExtractorFactory.create_model(ft_ext_param['user_ft_ext_param'], n_users,
                                                                          embedding_dim)
            item_feature_extractor = FeatureExtractorFactory.create_model(ft_ext_param['item_ft_ext_param'], n_items,
                                                                          embedding_dim)
            return user_feature_extractor, item_feature_extractor

        # For user or item based ProtoMF
        elif ft_type == 'prototypes':
            # The feature extractors are related, e.g. one of them contains a prototype layer and the other an embedding
            if 'prototypes' in ft_ext_param['user_ft_ext_param']['ft_type'] and \
                    ft_ext_param['item_ft_ext_param']['ft_type'] == 'embedding':
                # User Proto

                user_n_prototypes = ft_ext_param['user_ft_ext_param']['n_prototypes']

                user_feature_extractor = FeatureExtractorFactory.create_model(ft_ext_param['user_ft_ext_param'],
                                                                              n_users,
                                                                              embedding_dim)
                item_feature_extractor = FeatureExtractorFactory.create_model(ft_ext_param['item_ft_ext_param'],
                                                                              n_items,
                                                                              user_n_prototypes)

            elif 'prototypes' in ft_ext_param['item_ft_ext_param']['ft_type'] and \
                    ft_ext_param['user_ft_ext_param']['ft_type'] == 'embedding':
                # Item Proto
                item_n_prototypes = ft_ext_param['item_ft_ext_param']['n_prototypes']

                user_feature_extractor = FeatureExtractorFactory.create_model(ft_ext_param['user_ft_ext_param'],
                                                                              n_users,
                                                                              item_n_prototypes)
                item_feature_extractor = FeatureExtractorFactory.create_model(ft_ext_param['item_ft_ext_param'],
                                                                              n_items,
                                                                              embedding_dim)

            else:
                raise ValueError('Combination of ft_type of user/item feature extractors not valid for prototypes')

            return user_feature_extractor, item_feature_extractor
        
        # For coupled user and item based ProtoMF
        elif ft_type == 'prototypes_double_tie':
            # User-Item Proto
            item_n_prototypes = ft_ext_param['item_ft_ext_param']['n_prototypes']
            user_n_prototypes = ft_ext_param['user_ft_ext_param']['n_prototypes']
            user_use_weight_matrix = ft_ext_param['user_ft_ext_param']['use_weight_matrix']
            item_use_weight_matrix = ft_ext_param['item_ft_ext_param']['use_weight_matrix']

            assert not user_use_weight_matrix and not item_use_weight_matrix, 'Use Weight Matrix should be turned off to tie the weights!'

            # Building User Proto branch
            ft_ext_param['user_ft_ext_param']['ft_type'] = 'prototypes'
            user_proto = FeatureExtractorFactory.create_model(ft_ext_param['user_ft_ext_param'], n_users, embedding_dim)
            ft_ext_param['user_ft_ext_param']['ft_type'] = 'embedding_w'
            ft_ext_param['user_ft_ext_param']['out_dimension'] = item_n_prototypes
            user_embed = FeatureExtractorFactory.create_model(ft_ext_param['user_ft_ext_param'], n_users, embedding_dim)

            # Building Item Proto branch
            ft_ext_param['item_ft_ext_param']['ft_type'] = 'prototypes'
            item_proto = FeatureExtractorFactory.create_model(ft_ext_param['item_ft_ext_param'], n_items, embedding_dim)
            ft_ext_param['item_ft_ext_param']['ft_type'] = 'embedding_w'
            ft_ext_param['item_ft_ext_param']['out_dimension'] = user_n_prototypes
            item_embed = FeatureExtractorFactory.create_model(ft_ext_param['item_ft_ext_param'], n_items, embedding_dim)

            # Tying the weights together
            user_embed.embedding_layer.weight = user_proto.embedding_ext.embedding_layer.weight
            item_embed.embedding_layer.weight = item_proto.embedding_ext.embedding_layer.weight

            user_feature_extractor = ConcatenateFeatureExtractors(user_proto, user_embed, invert=False)
            item_feature_extractor = ConcatenateFeatureExtractors(item_proto, item_embed, invert=True)

            return user_feature_extractor, item_feature_extractor
        else:
            raise ValueError(f'FeatureExtractor <{ft_type}> Not Implemented..yet')

    @staticmethod
    def create_model(ft_ext_param: dict, n_objects: int, embedding_dim: int) -> FeatureExtractor:
        """
        Creates the specified FeatureExtractor model by reading the ft_ext_param. Currently available:
        - Embedding: represents objects by learning an embedding, A.K.A. Collaborative Filtering.
        - EmbeddingW: As Embedding but followed by a linear layer.
        - PrototypeEmbedding: represents an object by the similarity to the prototypes.

        :param ft_ext_param: parameters specific for the model type. ft_ext_param.ft_type is used for switching between
                models.
        :param embedding_dim: dimension of the final embeddings (how many latent features in the prototypes/MF matrices)
        :param n_objects: number of objects in the system
        """

        ft_type = ft_ext_param["ft_type"]

        print('--- Building FeatureExtractor model ---')

        if ft_type == 'embedding':
            max_norm = ft_ext_param['max_norm'] if 'max_norm' in ft_ext_param else None
            only_positive = ft_ext_param['only_positive'] if 'only_positive' in ft_ext_param else False
            model = Embedding(n_objects, embedding_dim, max_norm, only_positive)

        elif ft_type == 'embedding_w':
            max_norm = ft_ext_param['max_norm'] if 'max_norm' in ft_ext_param else None
            out_dimension = ft_ext_param['out_dimension'] if 'out_dimension' in ft_ext_param else None
            use_bias = ft_ext_param['use_bias'] if 'use_bias' in ft_ext_param else False
            model = EmbeddingW(n_objects, embedding_dim, max_norm, out_dimension, use_bias)

        elif ft_type == 'prototypes':
            n_prototypes = ft_ext_param['n_prototypes'] if 'n_prototypes' in ft_ext_param else None
            max_norm = ft_ext_param['max_norm'] if 'max_norm' in ft_ext_param else None
            sim_proto_weight = ft_ext_param['sim_proto_weight'] if 'sim_proto_weight' in ft_ext_param else 1.
            sim_batch_weight = ft_ext_param['sim_batch_weight'] if 'sim_batch_weight' in ft_ext_param else 1.
            reg_proto_type = ft_ext_param['reg_proto_type'] if 'reg_proto_type' in ft_ext_param else 'soft'
            reg_batch_type = ft_ext_param['reg_batch_type'] if 'reg_batch_type' in ft_ext_param else 'soft'
            cosine_type = ft_ext_param['cosine_type'] if 'cosine_type' in ft_ext_param else 'shifted'
            use_weight_matrix = ft_ext_param['use_weight_matrix'] if 'use_weight_matrix' in ft_ext_param else False

            model = PrototypeEmbedding(n_objects, embedding_dim, n_prototypes, use_weight_matrix, sim_proto_weight,
                                       sim_batch_weight, reg_proto_type, reg_batch_type, cosine_type, max_norm)
            
        else:
            raise ValueError(f'FeatureExtractor <{ft_type}> Not Implemented')

        print('--- Finished building FeatureExtractor model ---\n')
        return model




