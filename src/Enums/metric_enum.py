"""
Author: Mélanie Gaillochet
Date: 2020-10-27
"""
from Utils.metrics import mean_dice_metric, per_channel_mean_dice_metric, meanIoU

metrics = {
    'manual_dice': mean_dice_metric,
    'per_channel_dice': per_channel_mean_dice_metric,
    'meanIoU': meanIoU
}
