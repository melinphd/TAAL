"""
Author: Mélanie Gaillochet
Date: 2020-11-20
"""
from comet_ml import Experiment
import os
import copy
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch

from Base.base_solver import BaseSolver
from Utils.utils import add_dic_values, default0, to_onehot


class Solver(BaseSolver):
    """
    This solver performs one forward pass through the model with the labeled data
    """
    def __init__(self, config, test_dataloader, **kwargs):
        super().__init__(config, test_dataloader, **kwargs)
        
        print('Initializing Vanilla solver')

        self.loss_name_list = ['total', 'model']

    def train_step(self, data, target):
        """
        We implement the logic of one train step
        - return any metrics you need to summarize
        """
        
        # We set the model to training mode
        self.models_dic['model'].train()

        # We zero the parameter gradients
        self.optimizers_dic['model'].zero_grad()

        data, target = data.to(self.device, dtype=torch.float), target.to(self.device)

        #### Supervised part ####
        # We compute the model loss (supervised loss) using S(x)
        supervised_output, _ = self.models_dic['model'](data)
        onehot_target = to_onehot(target.squeeze(1), self.models_dic['model'].out_channels)
        model_loss = self.model_loss(supervised_output, onehot_target)
        train_loss = 1.0 * model_loss
        
        if self.activate_plot and (self.epoch % self.log_every == 0) and (self.train_batch_idx in [0, 1, 2]):
            self.plot_training_data_pred(data, target, supervised_output, indices=self.idx_l, type='Train')
            self.plot_training_data_pred_contour(data, target, supervised_output, indices=self.idx_l, type='Train')

        # We do backward propagation
        train_loss.backward()
        self.optimizers_dic['model'].step()

        loss_dic = {
            'total': train_loss.item(),
            'model': model_loss.item()
        }

        # We compute the metrics
        train_acc_dic = self.compute_metrics(supervised_output, onehot_target)

        return supervised_output, loss_dic, train_acc_dic

    def validation(self, val_dataloader, mode='val'):
        assert mode == 'val' or mode == 'test'
        
        # We set model in evaluation mode
        self.models_dic['model'].eval()

        val_dataloader.dataset.training = False

        # We initialize loss and accuracy
        val_acc_dic = defaultdict(default0)
        val_loss_dic = {}
        for key in self.loss_name_list:
            val_loss_dic[key] = 0

        # We iterate through validation batches
        with torch.no_grad():
            for batch_idx, (val_data, val_target, idx_list) in enumerate(val_dataloader):
                # print('val_batch {}, val_idx {}'.format(batch_idx, idx_list))
                val_data = val_data.to(self.device, dtype=torch.float)
                val_target = val_target.to(self.device)

                val_output, _ = self.models_dic['model'](val_data)

                #### Supervised loss ####
                # Computing the model loss (supervised loss) using S(x)
                onehot_target = to_onehot(val_target.squeeze(1), self.models_dic['model'].out_channels)
                model_loss = self.model_loss(val_output, onehot_target)
                total_loss = 1.0 * model_loss

                batch_val_loss_dic = {
                    'total': total_loss.item(),
                    'model': model_loss.item()
                }

                # We compute the metrics
                batch_val_acc_dic = self.compute_metrics(val_output, onehot_target)

                # We add the loss and metrics for each batch
                # Note for accuracy, we put the batch values first because initial val_acc_dic is empty
                val_loss_dic = add_dic_values(val_loss_dic, batch_val_loss_dic)
                val_acc_dic = add_dic_values(batch_val_acc_dic, val_acc_dic)

                if self.activate_plot and (mode == 'val') and (self.epoch % self.log_every == 0) and batch_idx in [0, 1, 2, 3]:
                    self.plot_val_data_pred(val_data, val_target, val_output, batch_idx, img_idx=0)
                    self.plot_val_data_pred_contour(val_data, val_target, val_output, batch_idx, img_idx=0)
                    
        # We compute average accuracy and loss over all batches
        avg_val_loss_dic = self.compute_mean_value(val_loss_dic, len(val_dataloader))
        avg_val_acc_dic = self.compute_mean_value(val_acc_dic, len(val_dataloader))

        return avg_val_loss_dic, avg_val_acc_dic, True
    