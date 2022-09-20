"""
Author: Mélanie Gaillochet
Date: 2020-10-27
"""

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import _LRScheduler

from Enums.scheduler_enum import schedulers


def create_scheduler(config, optimizer):
    """
    We make the learning rate scheduler
    :param config:
    :return:
    """
    scheduler_name = config["sched_name"]

    # We create an instance of the scheduler
    scheduler = schedulers[scheduler_name]

    if scheduler_name == 'StepLR':
        scheduler = scheduler(optimizer, step_size=config['step_size'], gamma=config['gamma'])
    elif scheduler_name == 'MultiStepLR':
        scheduler = scheduler(optimizer, milestones=config['milestones'], gamma=config['gamma'])
    elif scheduler_name == 'ReduceOnPlateau':
        scheduler = scheduler(optimizer, mode=config['mode'], factor=config['factor'],
                              patience=config['patience'], min_lr=config['min_lr'],
                              eps=config['eps'],
                              threshold=config['threshold'])
    elif scheduler_name == 'CosineAnnealingLR':
        scheduler = scheduler(optimizer, T_max=config["max_epoch"] - config["warmup_max"], eta_min=1e-7 )

    if config['gradual_warmup']:
        scheduler = GradualWarmupScheduler(optimizer, config["multiplier"],
                                            total_epoch=config["warmup_max"],
                                            after_scheduler=scheduler)
    return scheduler



"""from https://github.com/jizongFox/deepclustering2/blob/master/deepclustering2/schedulers/warmup_scheduler.py"""
class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier <= 1.0:
            raise ValueError("multiplier should be greater than 1.")
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [
                        base_lr * self.multiplier for base_lr in self.base_lrs
                    ]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        return [
            base_lr
            * ((self.multiplier - 1.0) * self.last_epoch / self.total_epoch + 1.0)
            for base_lr in self.base_lrs
        ]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = (
            epoch if epoch != 0 else 1
        )  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [
                base_lr
                * ((self.multiplier - 1.0) * self.last_epoch / self.total_epoch + 1.0)
                for base_lr in self.base_lrs
            ]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group["lr"] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)