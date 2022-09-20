"""
Author: Mélanie Gaillochet
Date: 2020-10-27
"""
from torch.optim import lr_scheduler

schedulers = {
    'StepLR': lr_scheduler.StepLR,
    'MultiStepLR': lr_scheduler.MultiStepLR,
    'ReduceOnPlateau': lr_scheduler.ReduceLROnPlateau,
    'CosineAnnealingLR': lr_scheduler.CosineAnnealingLR
    
}

"""
Variables to be defined in config
StepLR: {
    "step_size" (decreased LR every k epochs)
    "gamma" (factor by which LR will be reduced)
},

MultiStepLR: {
    "milestones"  (list of epochs where the LR is decreased)
    "gamma" (factor by which LR will be reduced)
},

ReduceOnPlateau: {
    "mode" (ie: "max", "min")  - LR reduced when quantity has reached max or min
    "factor" (default 0.1) -  Factor by which LR will be reduced
    "patience": (default 10) -   # epochs with no improvement after which LR reduced
    "verbose": (ie: True, False)
    "min_lr": (default 0)  - Lower bound on LR
    unchanged "threshold" (default 1e-4) -  Threshold for measuring the new optimum,
}
"""
