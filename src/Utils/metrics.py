"""
Author: Mélanie Gaillochet
Date: 2020-10-05
"""

"""
See https://github.com/PatrickChrist/LITS-CHALLENGE/blob/master/evaluation_notebook.ipynb
"""
# from medpy import metric
# from medpy import metric

import torch
import sklearn


def compute_dice(pred, tg, eps=1e-9, global_dice=False, reduction='mean'):
    """
    We compute the dice for a 3d image
    :param pred: normalized tensor (3d) [BS, x, y, z]
    :param target: tensor (3d) [BS, x, y, z]
    :param eps:
    :param normalize_fct:
    :param weighted:
    :return:
    """
    # if we compute the global dice then we will sum over the batch dim,
    # otherwise no
    if global_dice:
        dim = list(range(0, len(pred.shape)))
    else:
        dim = list(range(1, len(pred.shape)))

    intersect = torch.sum(pred * tg, dim=dim)
    union = pred.sum(dim=dim) + tg.sum(dim=dim)
    dice = (2. * intersect + eps) / (union + eps)

    if reduction == 'mean':
        # We average over the number of samples in the batch
        dice = dice.mean()

    return dice


def mean_dice_per_channel(predictions, onehot_target, eps=1e-9,
                          global_dice=False,
                          reduction='mean'):
    """
    We compute the dice, averaged for each channel
    :pa
    """

    dice_dic = {}
    for c in range(1, onehot_target.shape[1]):
        # We select only the predictions and target for the given class
        _selected_idx = torch.tensor([c])
        selected_idx = _selected_idx.to(predictions.get_device())
        pred = torch.index_select(predictions, 1, selected_idx)
        tg = torch.index_select(onehot_target, 1, selected_idx)

        # For each channel, we compute the mean dice
        dice = compute_dice(pred, tg, eps=eps, global_dice=global_dice,
                            reduction=reduction)
        dice_dic['dice_{}'.format(c)] = dice

    return dice_dic


def mean_dice_metric(onehot_pred, onehot_target, eps=1e-9, reduction='mean',
                     global_dice=False):
    """
    We compute the mean dice over every channel (except the first one which is the background)

    :param input: tensor of predictions (one hot for each channel) (BS, #classes, x, y, (z))
    :param target: target tensor (one hot for each channel) (BS, #classes, x, y, (z))
    :return:
    """
    # We make sure inputs are tensors of the same shape
    assert onehot_pred.shape == onehot_target.shape

    # We compute the per-channel dice (but we do not count the background)
    dice_dic = mean_dice_per_channel(onehot_pred, onehot_target, eps=eps,
                                     global_dice=global_dice, reduction=reduction)

    # We take the mean over all channels
    mean_dice = sum(dice_dic.values()) / len(dice_dic)

    return mean_dice


def per_channel_mean_dice_metric(onehot_pred, onehot_target, eps=1e-9,
                                 reduction='mean', global_dice=False):
    """
    We compute the mean dice for each channel (except the first one which is the background)

    :param input: tensor of predictions (one hot for each channel) (BS, #classes, x, y, (z))
    :param target: target tensor (one hot for each channel) (BS, #classes, x, y, (z))
    :return:
    """
    # We make sure inputs are tensors of the same shape
    assert onehot_pred.shape == onehot_target.shape

    # We compute the per-channel dice (but we do not count the background)
    dice_dic = mean_dice_per_channel(onehot_pred, onehot_target, eps=eps,
                                     global_dice=global_dice, reduction=reduction)

    return dice_dic


def interseOverUnion(onehot_pred, onehot_target, eps=1e-5):
    #outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
    
    if len(onehot_pred.shape) == 4:
        intersection = (onehot_pred & onehot_target).sum((-1, -2))  # Will be zero if Truth=0 or Prediction=0
        union = (onehot_pred | onehot_target).float().sum((-1, -2))         # Will be zzero if both are 0

    elif len(onehot_pred.shape) == 5:
        intersection = (onehot_pred & onehot_target).sum((-1, -2, -3))  # Will be zero if Truth=0 or Prediction=0
        union = (onehot_pred | onehot_target).float().sum((-1, -2, -3))         # Will be zzero if both are 0
    
    iou_score  = (intersection + eps) / (union + eps)  # We smooth our devision to avoid 0/0
    return iou_score


def meanIoU(onehot_pred, onehot_target, eps=1e-5):
    """ We compute the IoU for"""
    #squeezed_onehot_target = torch.squeeze(onehot_target)
    per_channels_iou = interseOverUnion(onehot_pred, onehot_target, eps)

    # Taking the mean over all channels
    _mean_iou_score = torch.mean(per_channels_iou, dim=1)

    # Taking the mean over alls amples of the batch
    mean_iou_score = torch.mean(_mean_iou_score, dim=0)
    return mean_iou_score