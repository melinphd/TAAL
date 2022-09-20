"""
Author: Mélanie Gaillochet
Date: 2020-11-18

"""
from comet_ml import Experiment
from Utils.utils import random_selection


class RandomSampler:
    """
    Random sampler
    """

    def __init__(self, budget):
        self.budget = budget

    def sample(self, unlabeled_dataloader):
        """ We randomly sample from the dataloader indices"""

        # We make the list of indices from the dataloader
        indice_list = []
        for _, _, index in unlabeled_dataloader:
            cur_index = index.cpu().numpy()
            indice_list.extend(cur_index.tolist())

        # We randomly select indices according to our budget
        num_selected = self.budget if len(indice_list) >= self.budget else len(
            indice_list)
        querry_pool_indices = random_selection(num_selected, indice_list)

        # We select the first value since random_selection outputs indices for 2 lists
        querry_pool_indices = querry_pool_indices[0]

        return querry_pool_indices
