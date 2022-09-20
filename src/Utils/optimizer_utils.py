"""
Author: Mélanie Gaillochet
Date: 2020-10-27
"""
from Enums.optimizer_enum import optimizers


def create_optimizer(config, model_params):
    """
    We create an optimizer based on the input config params
    :param optimizer:
    :param config:
    :return:
    """
    optimizer_name = config["optimizer_name"]

    # We create an instance of the optimizer
    optimizer = optimizers[optimizer_name]

    if optimizer_name == 'SGD':
        optimizer = optimizer(model_params, lr=config['init_lr'],
                              momentum=config['momentum'], weight_decay=config['weight_decay'])
    elif optimizer_name == 'Adam':
        optimizer = optimizer(model_params, lr=config['init_lr'], betas=config['betas'], weight_decay=config['weight_decay'])
    elif optimizer_name == 'RAdam':
        optimizer = optimizer(model_params, lr=config['init_lr'], betas=config['betas'], weight_decay=config['weight_decay'])

    return optimizer
