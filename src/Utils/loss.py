"""
Author: Mélanie Gaillochet
Date: 2020-10-13
"""
import torch.nn as nn
import torch.nn.functional as F

from Utils.metrics import mean_dice_per_channel
from Utils.utils import normalize


class DiceLoss(nn.Module):
    """
    This loss is based on the mean dice computed over all channels
    """

    def __init__(self, normalize_fct='softmax', reduction='mean'):
        super(DiceLoss, self).__init__()
        # print('Using {} normalization function'.format(normalize_fct))
        self.normalize_fct = normalize_fct
        self.reduction = reduction
        self.global_dice = False

    def forward(self, logits, onehot_target, eps=1e-9):
        pred = normalize(self.normalize_fct, logits)

        dice_dic = mean_dice_per_channel(pred, onehot_target,
                                         global_dice=self.global_dice,
                                         reduction=self.reduction, eps=eps)
        mean_dice = sum(dice_dic.values()) / len(dice_dic)

        loss = 1 - mean_dice

        return loss
