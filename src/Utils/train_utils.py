"""
Author: Mélanie Gaillochet
Date: 2021-05-31
"""

import random
import re
import time
import numpy as np
import torch
import torch.nn as nn

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, \
    classification_report
from Utils.utils import add_dic_values, default0, emptylist, add_dic_values, to_onehot, \
    find_best_target_slice, \
    find_worst_slice, emptylist, round_dic_values, convert_time, defaultinf, \
    logits_to_onehot, default0, normalize


def sigmoid_rampup(current, rampup_length):
    """
    Exponential rampup from https://arxiv.org/abs/1610.02242
    
    Code from 
    # Copyright (c) 2018, Curious AI Ltd. All rights reserved.
    #
    # This work is licensed under the Creative Commons Attribution-NonCommercial
    # 4.0 International License. To view a copy of this license, visit
    # http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
    # Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

    The sigmoid rampup is a Gaussian weighting function that slowly increases from close to 0 to 1
    :param current (int): current training step or epoch (int)
    :param rampup_length (int): rampup length defining the maximum step or epoch after which the function yields 1
    """
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

def softmax_with_temp(input, dim=1, t=1.0):
    """
    Softmax function with temperature
    """
    ex = torch.exp(input/t)
    print('ex {}'.format(ex.shape))
    sum_ex = torch.sum(ex, dim=dim)
    print('sum_ex {}'.format(sum_ex.shape))
    return ex / sum_ex


def compute_metrics(output, onehot_target, metric_dic, model_norm_fct, out_channels):
    """
    We create a dictionary with the metrics given in metric_enums
    :param output:
    :param onehot_target:
    :return:
    """
    acc_dic = {}
    for metric_name, cur_metric in metric_dic.items():
        # We convert the output logits to binary predictions for each channel
        onehot_pred = logits_to_onehot(output, model_norm_fct,
                                       out_channels)
        metric_value = cur_metric(onehot_pred, onehot_target)

        # If the metric value is a dictionary (ie: dice value for each channel),
        # then the accuracy dictionary will take in each entry separately
        if isinstance(metric_value, dict):
            for key, value in metric_value.items():
                acc_dic[key] = value.detach().cpu().numpy()
        # Otherwise, we will just add the current metric to the dictionary
        else:
            try:
                acc_dic[metric_name] = metric_value.detach().cpu().numpy()
            except AttributeError:
                acc_dic[metric_name] = metric_value
    return acc_dic


def compute_class_metrics(pred, target):
    """
    We create a dictionary with precision, recall, f1 score and accuracy metrics
    :param pred:
    :param target:
    :return:
    """
    metrics_dic = {
        'precision': precision_score(target, pred, average='macro', zero_division=0),
        'recall': recall_score(target, pred, average='macro', zero_division=0),
        'f1': f1_score(target, pred, average='macro', zero_division=0),
        'accuracy': accuracy_score(target, pred)
    }
    return metrics_dic


def compute_mean_value(input, num_items):
    """
    We compute the mean value of a float or a dictionary
    :return:
    """
    if isinstance(input, float):
        mean_input = input / num_items

    elif isinstance(input, dict):
        mean_input = {}
        for key in input:
            mean_input[key] = input[key] / num_items
    return mean_input


def print_epoch_update(epoch, train_loss_dic, val_loss_dic, model_train_acc_dic,
                       model_val_acc_dic, epoch_start_time,
                       lr_dic, best_losses_dic='', best_model_val_acc=''):
    """
    We print the train and validation losses/metrics
    """
    # We round up all the values
    nice_train_loss_dic = round_dic_values(train_loss_dic, 4)
    nice_model_train_acc_dic = round_dic_values(model_train_acc_dic, 4)
    nice_val_loss_dic = round_dic_values(val_loss_dic, 4)
    nice_model_val_acc_dic = round_dic_values(model_val_acc_dic, 4)
    epoch_end_time = time.time()
    minutes, sec = convert_time(epoch_end_time - epoch_start_time)

    if best_losses_dic != '' and best_model_val_acc != '':
        print('Epoch {} - LR: {} - Train losses: {},  Train acc: {} \n'
              'Val loss: {} (model best: {:.4f}), Val acc: {} (best: {:.4f})  \n -  '
              'Time taken: {}min, {}sec'.format(epoch, lr_dic,
                                                nice_train_loss_dic, nice_model_train_acc_dic,
                                                nice_val_loss_dic, best_losses_dic['model'],
                                                nice_model_val_acc_dic, best_model_val_acc,
                                                minutes, sec))
    else:
        print('Epoch {} - LR: {} - Train losses: {},  Train acc: {} \n'
              'Val loss: {}, Val acc: {}  \n -  '
              'Time taken: {}min, {}sec'.format(epoch, lr_dic,
                                                nice_train_loss_dic, nice_model_train_acc_dic,
                                                nice_val_loss_dic, nice_model_val_acc_dic,
                                                minutes, sec))


def apply_dropout(module):
    """
    This function activates dropout modules. Dropout module should have been defined as nn.Dropout
    Args:
        m ([type]): [description]
    """
    if type(module) == nn.Dropout:
        module.train()

