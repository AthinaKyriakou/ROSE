import torch
import random

# This class ensures easy access to adjust standard parameters in the ProtoMF model

# Standard parameters used in all model variations
class Std_params():
    N_EPOCHS = 25
    K = 10
    LEARNING_RATE = 0.01
    WEIGHT_DECAY = 1e-4
    OPTIM_NAME = "adam"
    LOSS = "bce"
    MAX_PATIENCE = 5
    

# Standard parameter separately for each model variation
def get_ft_ext_param(model):
    """
    This function serves as a helper to get all the needed parameters depending on which model was chosen
    Args:
        model - string, specifies the model (selection of mf, user_proto, item_proto or user_item_proto)
    Returns:
        dictionary containing all necessary parameters
    """
    base_param = {
    'n_epochs': 100,
    'eval_neg_strategy': 'uniform',
    'val_batch_size': 256,
    'rec_sys_param': {'use_bias': 0},
    }


    if model == 'mf':
        return {**base_param,
                'name': "ProtoMF - MF",
                "ft_type": "detached",
                'embedding_dim': 10,
                'user_ft_ext_param': {"ft_type": "embedding",},
                'item_ft_ext_param': {"ft_type": "embedding",}
                }
    
    if model == 'user_proto':
        return {**base_param,
                'name': "ProtoMF - User",
                'loss_func_aggr': 'mean',
                "ft_type": "prototypes",
                'embedding_dim': random.randint(10, 100),
                'user_ft_ext_param': {
                    "ft_type": "prototypes",
                    'sim_proto_weight': random.uniform(1e-3, 10),
                    'sim_batch_weight': random.uniform(1e-3, 10),
                    'use_weight_matrix': False,
                    'n_prototypes': random.randint(10, 100),
                    'cosine_type': 'shifted',
                    'reg_proto_type': 'max',
                    'reg_batch_type': 'max',
                    },
                'item_ft_ext_param': {
                    "ft_type": "embedding",
                }
            }
    if model == 'item_proto':
        return {**base_param,
                'name': "ProtoMF - Item",
                'loss_func_aggr': 'mean',
                "ft_type": "prototypes",
                'embedding_dim': random.randint(10, 100),
                'item_ft_ext_param': {
                    "ft_type": "prototypes",
                    'sim_proto_weight': random.uniform(1e-3, 10),
                    'sim_batch_weight': random.uniform(1e-3, 10),
                    'use_weight_matrix': False,
                    'n_prototypes': random.randint(10, 100),
                    'cosine_type': 'shifted',
                    'reg_proto_type': 'max',
                    'reg_batch_type': 'max'
                },
                'user_ft_ext_param': {
                    "ft_type": "embedding",
                }
            }
    
    if model == 'user_item_proto':
        return {**base_param,
                'name': "ProtoMF - User&Item",
                'loss_func_aggr': 'mean',
                "ft_type": "prototypes_double_tie",
                'embedding_dim': random.randint(10, 100),
                'item_ft_ext_param': {
                    "ft_type": "prototypes_double_tie",
                    'sim_proto_weight': random.uniform(1e-3, 10),
                    'sim_batch_weight': random.uniform(1e-3, 10),
                    'use_weight_matrix': False,
                    'n_prototypes': random.randint(10, 100),
                    'cosine_type': 'shifted',
                    'reg_proto_type': 'max',
                    'reg_batch_type': 'max'
                },
                'user_ft_ext_param': {
                    "ft_type": "prototypes_double_tie",
                    'sim_proto_weight': random.uniform(1e-3, 10),
                    'sim_batch_weight': random.uniform(1e-3, 10),
                    'use_weight_matrix': False,
                    'n_prototypes': random.randint(10, 100),
                    'cosine_type': 'shifted',
                    'reg_proto_type': 'max',
                    'reg_batch_type': 'max'
                }
            }
    raise ValueError("Unknown model, please choose one of the following options: mf, user_proto, item_proto or user_item_proto")
