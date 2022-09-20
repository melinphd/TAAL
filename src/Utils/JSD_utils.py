"""
Author: Mélanie Gaillochet
Date: 2022-03-25
"""

import torch

from Utils.utils import normalize
from Utils.augmentation_utils import random_train_transforms, reverse_geometry_transform


def make_aug_data_list(unsup_data, num_augmentations, augmentation_list, augmentation_params, device):
    """ We generate T1(x'), T2(x'), T3(x')
    Args:
        unsup_data (tensor): unsupervised data which we want to augment
    Returns:
        (list of tensors) aug_data: list of augmented data (length=self.consistency_num_augmentations, each with shape (BS, C, H, W, ..)) 
        (list of dics) param_dic_list: list of augmentation parameters (length=self.consistency_num_augmentations)
    """
    aug_data = []
    param_dic_list = []
    with torch.no_grad():
        for i in range(num_augmentations):
            # We augment the data
            transformed_data, _, param_dic = random_train_transforms(
                unsup_data, targets=None, augmentation_list=augmentation_list,
                augmentation_params=augmentation_params)
            transformed_data = transformed_data.to(device)
            
            aug_data.append(transformed_data)
            param_dic_list.append(param_dic)
    return aug_data, param_dic_list


def make_reverse_aug_prob_list(u_prob, aug_scores, param_dic_list, model_norm_fct):
    """ We generate T1^(-1)[S(T1(x'))], T2^(-1)[S(T2(x'))], T3^(-1)[S(T3(x'))], ...

    Args:
        u_prob (tensor): probability output of x' (untransformed unlabeled data)
        aug_scores (list of tensors): output logits of all T(x')
        param_dic_list (list of dics): list of all parameters for applied transforms

    Returns:
        (list of tensors) cur_prob_list: list of probability tensors after inverse transform
    """
    cur_prob_list = [u_prob]
    for i in range(len(param_dic_list)):
        # We do the inverse transformation on the output
        rev_aug_scores = reverse_geometry_transform(aug_scores[i * len(u_prob):(i+1) * len(u_prob)], param_dic_list[i])
        # We get output probability and prediction
        rev_aug_prob = normalize(model_norm_fct, rev_aug_scores)
        # We keep track of the output probabilities
        cur_prob_list.append(rev_aug_prob)
    return cur_prob_list


def make_reverse_aug_mask(data_shape, param_dic_list, num_augmentations):
    """ We generate a mask that covers only pixels which appear in all transformed versions of x' 
    (which do not disappear because of a rotation, for example)

    Args:
        data_shape (tupe): shape of data x' (ie: BS, C, H, W, ...)
        param_dic_list (list of dics): list of augmentation parameters (length=self.consistency_num_augmentations)

    Returns:
        (tensor) mask: True-False mask
    """
    _mask = torch.ones(data_shape)
    for i in range(num_augmentations):
        identity = torch.ones(data_shape)
        rev_identity = reverse_geometry_transform(
            identity, param_dic_list[i])
        # We update the mask (will be 1 whenever both tensor have value 1)
        _mask = torch.logical_and(_mask, rev_identity)
    mask = torch.unsqueeze(_mask, 1)
    mask = _mask.repeat(1, num_augmentations + 1, 1, 1, 1)
    return mask
