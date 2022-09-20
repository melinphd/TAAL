"""
Author: Mélanie Gaillochet
Date: 2020-10-05
"""
import os
import random
import re

import numpy as np
import torch
import json


def set_all_seed(seed=42):
    """
    We are setting all seeds
    :param seed: (default: 42)
    :return:
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def print_args(args):
    """
    We are printing all the input arguments and their value
    :param args:
    :return:
    """
    print('\n')
    for arg in vars(args):
        print('{} = {}'.format(arg, getattr(args, arg)))
    print('\n')


def _atoi(text):
    """
    We return the string as type int if it represents a number (or the string itself otherwise)
    """
    return int(text) if text.isdigit() else text


def natural_keys(text):
    """
    We return the list of string and digits as different entries,
    following the order in which they appear in the string
    """
    return [_atoi(c) for c in re.split('(\d+)', text)]


def convert_time(time):
    """
    We convert time value to min and seconds
    :param time:
    :return:
    """
    time = int(time)
    min = time // 60
    sec = time - min * 60

    return min, sec


def add_dic_values(dic1, dic2):
    """
    We add the value of dic2 to the value of dic1 for all keys of dic1
    :param dic1:
    :param dic2:
    :return: dic1 with updated values if there are some common keys with dic2
    """
    dic1_copy = dic1.copy()
    for key in dic1_copy:
        if key in dic2:
            dic1_copy[key] = dic1_copy[key] + dic2[key]
        else:
            pass
    return dic1_copy


def round_dic_values(dic, num_dec=6):
    """
    We want to round all values of dictionary to k decimals
    :param dic: dictinary with numbers as values
    :return:
    """
    for key, value in dic.items():
        dic[key] = np.round(value, num_dec)

    return dic


def to_onehot(input, n_classes):
    """
    We do a one hot encoding of each label in 3D.
    (ie: instead of having a dimension of size 1 with values 0-k,
    we have 3 axes, all with values 0 or 1)
    :param input: tensor
    :param n_classes:
    :return:
    """
    assert torch.is_tensor(input)

    # We get (bs, l, h, w, n_channels), where n_channels is now > 1
    one_hot = torch.nn.functional.one_hot(input, n_classes)

    # We permute axes to put # channels as 2nd dim
    if len(one_hot.shape) == 5:
        one_hot = one_hot.permute(0, 4, 1, 2, 3)
    elif len(one_hot.shape) == 4:
        one_hot = one_hot.permute(0, 3, 1, 2)
    return one_hot


def normalize(normalize_fct, x):
    """We apply sigmoid or softmax on the logits to get probabilities"""
    # The logits should be normalized. We are using softmax
    if normalize_fct == 'softmax':
        fct = torch.nn.Softmax(dim=1)

    elif normalize_fct == 'sigmoid':
        fct = torch.nn.Sigmoid()

    return fct(x)


def logits_to_onehot(logits, normalize_fct, num_classes):
    """
    We convert logits to one hot predictions
    :param
    """
    # We convert logits into probabilities
    predictions = normalize(normalize_fct, logits)

    # We take the highest probability to make the predictions
    labels_pred = torch.argmax(predictions, dim=1)

    # We convert the predictions to one hot predictions for each class
    onehot_pred = to_onehot(labels_pred, num_classes)

    return onehot_pred


def find_best_target_slice(data):
    """
    We want to find the slice across the z axis which gives the highest sum
    :param data: torch tensor (x, y, z)
    :return:
    """
    assert torch.is_tensor(data)

    best_slice = 0
    best_value = 0
    for i in range(0, data.shape[-1]):
        sum_slice = torch.sum(data[:, :, i])
        if sum_slice > best_value:
            best_slice = i
            best_value = sum_slice

    return best_slice


def find_worst_slice(pred, target):
    """
    We want to fiind the slice where we have the biggest error between our target and the prediction
    :param pred: torch tensor (x, y, z)
    :param target: torch tensor (x, y, z)
    :return:
    """

    assert torch.is_tensor(pred)
    assert torch.is_tensor(target)

    worst_slice = 0
    worst_diff = 0
    for i in range(0, target.shape[-1]):
        sum_target_slice = torch.sum(target[:, :, i])
        sum_pred_slice = torch.sum(pred[:, :, i])
        diff = torch.abs(sum_target_slice - sum_pred_slice)
        if diff > worst_diff:
            worst_slice = i
            worst_diff = diff

    return worst_slice


def default0():
    """ Function to be called by defaultdic to default values to 0 """
    return 0


def emptylist():
    """ Function to be called by defaultdic to default values to an empty list """
    return []


def defaultinf():
    """ Function to be called by defaultdic to default values to inf """
    return np.inf


def random_selection(num_items, x_list, y_list=None):
    """
    We are randomly selecting k items from a list (or 2 lists)
    :param num_items: number of items to select (either an integer or string 'all')
    :param x_list: list to select items from
    :param y_list: (optional) second list paired to first, to select items from
    :return:
    """
    if num_items == 'all':
        num_items = len(x_list)

    # We create a list of indices taken randomly
    random_idx = random.sample(range(len(x_list)), num_items)

    # We order the lists
    # x_list = x_list.sort(key=natural_keys)
    # y_list = y_list.sort(key=natural_keys) if y_list is not None else None

    # We select a list of items given the random indices
    rand_x_list = [x_list[idx] for idx in random_idx]

    # We select the corresponding items from y_list, if it is given
    rand_y_list = None
    if y_list is not None:
        assert len(x_list) == len(y_list)
        rand_y_list = [y_list[idx] for idx in random_idx]

    return rand_x_list, rand_y_list


def divide_round_up(n, d):
    """
    We do a division (n/d) but round up the number instead of truncating it
    :param n:
    :param d:
    :return:
    """
    return (n + (d - 1)) / d


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)