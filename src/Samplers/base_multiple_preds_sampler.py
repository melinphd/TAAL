"""
Author: Mélanie Gaillochet
Date: 2022-02-209

"""
from comet_ml import Experiment
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch

from Utils.sampler_utils import get_uncertainty_multiple_preds


class BaseMultiplePredsSampler:
    """
    Base sampler for based on uncertainty from different predicted results
    """

    def __init__(self, budget):
        self.budget = budget

    def base_sample(self, model, unlabeled_dataloader, device, sampling_type, transformation_type, num_dropout_inference=None, alpha_jsd=0.5,
                    data_aug_gaussian_mean=0, data_aug_gaussian_std=0):
        
        indice_list, mean_variance_list, max_variance_list, mean_jsd_list = get_uncertainty_multiple_preds(
            model, unlabeled_dataloader, device, transformation_type, num_dropout_inference, alpha_jsd, data_aug_gaussian_mean, data_aug_gaussian_std)
        
        if 'MeanVariance' in sampling_type:
            uncertainty = mean_variance_list
        elif 'MaxVariance' in sampling_type:
            uncertainty = max_variance_list
        elif 'MeanJSD' in sampling_type:
            uncertainty = mean_jsd_list

        # Index in ascending order
        arg = np.argsort(uncertainty)
        querry_pool_indices = list(torch.tensor(indice_list)[arg][-self.budget:].numpy())
        uncertainty_values = list(torch.tensor(uncertainty)[arg][-self.budget:].numpy())

        return querry_pool_indices, uncertainty_values

