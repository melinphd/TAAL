"""
Author: Mélanie Gaillochet
Date: 2021-03-25
"""
import os
import numpy as np
import torch
import torch.utils.data as data

from Utils.sampler_utils import InfiniteSubsetRandomSampler, SubsetSequentialSampler


def update_kwargs(kwargs, method, config, sampling_type, labeled_indices=None, unlabeled_indices=None,
                  unlabeled_dataloader=None, dataset=None, batch_size=None, seed=42):
    """
    This function updates the variables/params needed according to the AL method used
    :param kwargs: kwargs that we want to update
    :param method: AL method used
    :param config: config file used
    :param labeled_indices: (needed for VAAL and IMSAT_detach_unsup)
    :param unlabeled_indices: (needed for VAAL and IMSAT_detach_unsup)
    :param unlabeled_dataloader: (needed for VAAL)
    :param dataset: (possibly augmented) dataset
    :param batch_size: (needed for VAAL, VAE and ACNN)
    :param debug: (needed for VAAL and VAE and)
    :param seed: (needed for VAAL, VAE and ACNN)
    """

    if 'Coresets' in sampling_type:
        finite_seq_labeled_sampler = SubsetSequentialSampler(labeled_indices)
        kwargs['finite_labeled_dataloader'] = data.DataLoader(dataset, sampler=finite_seq_labeled_sampler, batch_size=1,
                                                              drop_last=False, num_workers=config['training']['num_workers'],
                                                          persistent_workers=True,
                                                          pin_memory=True)
        
    if 'SemiSupervised' in method:
        extra_sampler = InfiniteSubsetRandomSampler(dataset, labeled_indices + unlabeled_indices, shuffle=True)
        extra_dataloader = data.DataLoader(dataset, sampler=extra_sampler,
                                            batch_size=batch_size, drop_last=False, pin_memory=True,
                                            num_workers=config['training']['num_workers'])
        kwargs['extra_dataloader'] = extra_dataloader

    return kwargs
