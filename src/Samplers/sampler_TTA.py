"""
Author: Mélanie Gaillochet
Date: 2021-12-23

"""
from Samplers.base_multiple_preds_sampler import BaseMultiplePredsSampler


class TestTimeAugmentationSampler(BaseMultiplePredsSampler):
    """
    Sampler for test time augmentation (based on difference between the different results)
    """

    def __init__(self, budget):
        self.budget = budget

    def sample(self, model, unlabeled_dataloader, device, experiment, sampling_type, alpha_jsd, data_aug_gaussian_mean, data_aug_gaussian_std):
        
        querry_pool_indices, uncertainty_values = self.base_sample(model, unlabeled_dataloader, device, experiment, sampling_type, 'data_aug', alpha_jsd=alpha_jsd,
                                                                   data_aug_gaussian_mean=data_aug_gaussian_mean, data_aug_gaussian_std=data_aug_gaussian_std)

        return querry_pool_indices, uncertainty_values
