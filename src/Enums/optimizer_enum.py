"""
Author: Mélanie Gaillochet
Date: 2020-10-27
"""
import torch.optim as optim

optimizers = {
    'SGD': optim.SGD,
    'Adam': optim.Adam,
    'RAdam': optim.RAdam
}

"""
Variables to be defined in config
SGD: {
    "init_lr"
    "momentum" (default 0)
    "weight_decay" (default 0)
}
Adam: {
    "init_lr" (default 0.001)
    "betas" (default (0.9, 0.999))
    unchanged "eps" (default 1e-8)
    unchanged "weight decay" (default 0)
}
"""
