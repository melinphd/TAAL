"""
Author: Mélanie Gaillochet
Date: 2021-11-26
"""
from collections import Iterator
import numpy as np

import torch
from torch.nn import functional as F
from torch.utils.data import Sampler

from Utils.train_utils import apply_dropout
from Utils.augmentation_utils import augment_data, reverse_augment_data
from Utils.augmentation_utils import entropy_2d, JSD


class _InfiniteSubsetRandomIterator(Iterator):
    def __init__(self, data_source, indices, shuffle=True):
        self.data_source = data_source
        self.indices = indices
        self.shuffle = shuffle
        if self.shuffle:
            permuted_indices = np.random.permutation(self.indices).tolist()
            self.iterator = iter(permuted_indices)
        else:
            self.iterator = iter(self.indices)

    def __next__(self):
        try:
            idx = next(self.iterator)
        except StopIteration:
            if self.shuffle:
                permuted_indices = np.random.permutation(self.indices).tolist()
                self.iterator = iter(permuted_indices)
            else:
                self.iterator = iter(self.indices)
            idx = next(self.iterator)
        return idx


class InfiniteSubsetRandomSampler(Sampler):
    """
    This is a sampler that randomly selects data indices from a list of indices, in an infinite loop
    """

    def __init__(self, data_source, indices, shuffle=True):
        self.data_source = data_source
        self.indices = indices
        self.shuffle = shuffle

    def __iter__(self):
        return _InfiniteSubsetRandomIterator(self.data_source, self.indices,
                                             shuffle=self.shuffle)

    def __len__(self):
        return len(self.data_source)


class SubsetSequentialSampler(torch.utils.data.Sampler):
    r"""Samples elements sequentially from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)


class InfiniteRandomSampler(Sampler):

    def __init__(self, data_source, shuffle=True):
        super().__init__(data_source)
        self.data_source = data_source
        self.shuffle = shuffle

    def __iter__(self):
        if len(self.data_source) > 0:
            while True:
                yield from self.__iter_once__()
        else:
            yield from iter([])

    def __iter_once__(self):
        if self.shuffle:
            return iter(torch.randperm(len(self.data_source)).tolist())
        return iter(torch.arange(start=0, end=len(self.data_source)).tolist())

    def __len__(self):
        return len(self.data_source)


def get_different_preds(transformation_type, inputs, model, device, num_dropout_inference=None, data_aug_gaussian_mean=0, data_aug_gaussian_std=0):
    """
    We get the different model predictions under the given transformation type

    :param transformation type: type of trasformation applied to data or model ('dropout' or 'data_aug')
    
    Output:
    - cur_prob_list (list of torch tensors)
    - cur_pred_list (list of numpy arrays)
    """
    model = model.to(device)
    model.eval()

    cur_pred_list = []
    cur_prob_list = []

    if transformation_type == 'data_aug':
        for flip in [0, 1]:
            for n_rotation in range(0, 4):
                # We do inference on the augmented data
                aug_inputs = augment_data(
                    inputs, flip, n_rotation, flip_axis=2, rot_axis0=2,
                    rot_axis1=3, mean_gaussian=data_aug_gaussian_mean, std_gaussian=data_aug_gaussian_std)
                aug_inputs = aug_inputs.to(device)
                aug_output, _ = model(aug_inputs)
                
                # We do the inverse transformation on the output
                rev_output = reverse_augment_data(aug_output, flip, n_rotation, flip_axis=2, rot_axis0=2, rot_axis1=3)
                
                # We get output probability and prediction
                rev_prob = F.softmax(rev_output, dim=1)
                rev_pred = torch.argmax(rev_prob, dim=1)
                
                # We keep track of the output probabilities and prediction
                cur_prob_list.append(rev_prob)
                cur_pred_list.append(rev_pred.detach().cpu().numpy())

                
    elif transformation_type == 'dropout':
        # We enable dropout
        model.apply(apply_dropout)

        for i in range(num_dropout_inference):
            cur_output, _ = model(inputs)
            
            # We get output probability and prediction
            cur_prob = F.softmax(cur_output, dim=1)
            cur_pred = torch.argmax(cur_prob, dim=1)
            
            # We keep track of the output probabilities and prediction
            cur_prob_list.append(cur_prob)
            cur_pred_list.append(cur_pred.detach().cpu().numpy())

    return cur_prob_list, cur_pred_list


def get_uncertainty_multiple_preds(model, unlabeled_dataloader, device, transformation_type, num_dropout_inference=None, alpha_jsd=0.5,
                                   data_aug_gaussian_mean=0, data_aug_gaussian_std=0):
    """
    We get the uncertainty measure for different predictions of the same unlabeled data 
    (ie: under different augmentations, difefrent dropout, etc.)
    The uncertainty is given as mean variance, max variance, mean entropy (averaged for all channels, then all pixels) and JSD
    """
    indice_list  = []
    data_list = []
    target_list = []
    preds_list = []
    variance_list = []
    entropy_list = []
    jsd_list = []
    mean_pred_list = []
    mean_variance_list = []
    max_variance_list = []
    mean_entropy_list = []
    mean_jsd_list = []

    # We iterate through the unlabeled datatloader
    with torch.no_grad():
        for (inputs, labels, index) in unlabeled_dataloader:
            inputs = inputs.to(device, dtype=torch.float)
            labels = labels.to(device)
            
            cur_prob_list, cur_pred_list =  get_different_preds(transformation_type, inputs, model, device, num_dropout_inference,
                                                                data_aug_gaussian_mean=data_aug_gaussian_mean, 
                                                                data_aug_gaussian_std=data_aug_gaussian_std)

            # We concatenate the output probability list of augmented inputs
            prob_concat = torch.concat(cur_prob_list, dim=0)
            #3print('prob_concat {}'.format(prob_concat.shape))

            # We compute the mean, variance, entropy and JSD of the result probabilities
            mean_prob = torch.mean(prob_concat, dim=0)  # Gives mean output probability (shape C x H x W)
            var_prob = torch.var(prob_concat, dim=0)  # Gives variance of output probabilities (shape C x H x W)
            entropy = entropy_2d(prob_concat, dim=0)   # Gives entropy over output probabilities (shape C x H x W)
            jsd = JSD(prob_concat, alpha=alpha_jsd)

            # We keep track of all dataloader results
            cur_index = index.cpu().numpy()
            indice_list.extend(cur_index.tolist())
            
            data_list.append(inputs.detach().cpu().numpy())
            target_list.append(labels.detach().cpu().numpy())
            preds_list.append(np.concatenate(cur_pred_list, axis=0))
            # We compute the mean prediction (shape H x W)
            mean_pred = torch.argmax(mean_prob, dim=0)
            mean_pred_list.append(mean_pred.detach().cpu().numpy())

            # We keep track of the variance
            image_variance = torch.mean(var_prob, dim=0)   # mean over all channels (shape H x W)
            variance_list.append(image_variance.detach().cpu().numpy())
            mean_var = torch.mean(var_prob)   # mean over all image (float)
            mean_variance_list.append(mean_var.item())
            max_var = torch.max(var_prob)  # mean over all image (float)
            max_variance_list.append(max_var.item())

            # We keep track of the entropy
            image_entropy = torch.mean(entropy, dim=0)   # mean over all channels (shape H x W)
            entropy_list.append(image_entropy.detach().cpu().numpy())
            mean_entropy = torch.mean(entropy)   # mean over all image (float)
            mean_entropy_list.append(mean_entropy.item())

            # We keep track of the JSD
            jsd_list.append(jsd.detach().cpu().detach())   # (shape H x W)
            mean_jsd = torch.mean(jsd)    # mean over all image (float)
            mean_jsd_list.append(mean_jsd.item())

    return indice_list, mean_variance_list, max_variance_list, mean_jsd_list