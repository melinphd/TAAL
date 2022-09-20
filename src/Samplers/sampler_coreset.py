"""
Author: Mélanie Gaillochet
Date: 2020-11-18

"""
from comet_ml import Experiment

import numpy as np
from sklearn.metrics import pairwise_distances

import torch
from torch.utils.data import DataLoader

from Utils.unet_utils import max_pooling_2d


class CoresetsSampler:
    """
    Coresets sampler
    Implementation adapted from https://github.com/anonneurips-8435/Active_Learning/blob/eba1acddf0eeddabce3ee618349369e89c4f31dd/main/active_learning_strategies/core_set.py
    A diversity-based approach using coreset selection. The embedding of each example is computed by the network’s 
    penultimate layer and the samples at each round are selected using a greedy furthest-first 
    traversal conditioned on all labeled examples.
    """

    def __init__(self, budget):
        self.budget = budget

    def sample(self, model, unlabeled_dataloader, device, experiment, labeled_dataloader, pooling_kwargs):
        # We put the models on GPU
        model = model.to(device)

        embedding_unlabeled, idx_unlabeled = get_embedding(model, unlabeled_dataloader, pooling_kwargs, device)
        embedding_labeled, idx_labeled = get_embedding(model, labeled_dataloader, pooling_kwargs, device)
        
        chosen_indices = furthest_first(embedding_unlabeled, embedding_labeled, self.budget)
        print('chosen_indices {}'.format(chosen_indices))
        
        querry_pool_indices = [idx_unlabeled[idx] for idx in chosen_indices]

        return querry_pool_indices


def furthest_first(unlabeled_set, labeled_set, budget):
    """
    Selects points with maximum distance
    
    Parameters
    ----------
    unlabeled_set: numpy array
        Embeddings of unlabeled set
    labeled_set: numpy array
        Embeddings of labeled set
    budget: int
        Number of points to return
    Returns
    ----------
    idxs: list
        List of selected data point indexes with respect to unlabeled_x
    """
    m = np.shape(unlabeled_set)[0]
    if np.shape(labeled_set)[0] == 0:
        min_dist = np.tile(float("inf"), m)
    else:
        dist_ctr = pairwise_distances(unlabeled_set, labeled_set)
        min_dist = np.amin(dist_ctr, axis=1)

    idxs = []

    for i in range(budget):
        idx = min_dist.argmax()
        idxs.append(idx)
        dist_new_ctr = pairwise_distances(
            unlabeled_set, unlabeled_set[[idx], :])
        for j in range(m):
            min_dist[j] = min(min_dist[j], dist_new_ctr[j, 0])

    return idxs


def get_embedding(model, dataloader, pooling_kwargs, device):
    model.eval()
    #embedding = torch.zeros([dataloader.shape[0], model.get_embedding_dim()])
    embedding_list = []
    idx_list = []

    pool = max_pooling_2d(**pooling_kwargs)

    with torch.no_grad():
        for data, _, idxs in dataloader:
            data = data.to(device, dtype=torch.float)
            _, [enc_1, enc_2, enc_3, center, dec_1, dec_2, dec_3] = model(data)
            ##embedding[idxs] = features.data.cpu()
            #cur_features = dec_3.view(dec_3.size(0), -1)

            pooled_dec = pool(dec_3)
            cur_features = pooled_dec.view(pooled_dec.size(0), -1)
            embedding_list.append(cur_features.data.cpu())
            idx_list.append(idxs.item())
    embedding = np.concatenate(embedding_list, axis=0)
    print('embedding {}'.format(embedding.shape))
    print(idx_list)
    return embedding, idx_list
 