"""
Author: Mélanie Gaillochet
Date: 2020-11-18

"""
from Samplers.base_multiple_preds_sampler import BaseMultiplePredsSampler


class DropoutSampler(BaseMultiplePredsSampler):
    """
    Sampler for dropout (based on difference between the different results)
    """

    def __init__(self, budget):
        self.budget = budget

    def sample(self, model, unlabeled_dataloader, device, experiment, sampling_type, num_dropout_inference, alpha_jsd):
        
        querry_pool_indices, uncertainty_values = self.base_sample(model, unlabeled_dataloader, device, experiment, sampling_type, 'dropout', num_dropout_inference=num_dropout_inference, alpha_jsd=alpha_jsd)

        return querry_pool_indices, uncertainty_values
