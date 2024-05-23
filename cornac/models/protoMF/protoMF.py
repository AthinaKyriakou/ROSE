from .feature_extractors import FeatureExtractorFactory

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

class ProtoMFModel(nn.Module):
    """
    This class represents our ProtoMF model which extends a torch module, meaning we can implement
    our own forward pass
    """
    def __init__(self, n_users, n_items, user_feature_extractor, item_feature_extractor):
        super().__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.user_feature_extractor = user_feature_extractor
        self.item_feature_extractor = item_feature_extractor

        # Initialize feature extractors:
        self.user_feature_extractor.init_parameters()
        self.item_feature_extractor.init_parameters()


    def forward(self, u_idxs, i_idxs):
        """
        Performs the forward pass considering user indexes and the item indexes. Negative Sampling is done automatically
        by the dataloader

        Parameters
        ----------
        u_idxs: array
            User indexes. Shape is (batch_size,)
        i_idxs: array
            Item indexes. Shape is (batch_size, n_neg + 1)
            
        Returns
        -------
        """
 
        # Embedd users and items accordingly
        u_embed = self.user_feature_extractor(u_idxs)
        i_embed = self.item_feature_extractor(i_idxs)
                 
        # Forward pass
        out = torch.sum(u_embed.unsqueeze(1) * i_embed, dim=-1)  # [batch_size, n_neg_p_1]
        return out