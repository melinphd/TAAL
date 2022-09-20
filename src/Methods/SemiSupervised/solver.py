"""
Author: Mélanie Gaillochet
Date: 2022-02-14

"""

import os
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import copy
import random

from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn import functional as F

from Base.base_solver import BaseSolver
from Utils.metrics import meanIoU
from Utils.utils import default0, add_dic_values, to_onehot
from Utils.augmentation_utils import random_augmentation, JSD, augment_data, reverse_augment_data
from Utils.train_utils import sigmoid_rampup


class Solver(BaseSolver):
    """
    This is the solver class for Cross-Augmentation Consistency training
    """

    def __init__(self, config, test_dataloader, **kwargs):
        super().__init__(config, test_dataloader, **kwargs)
        
        print('Initializing CrossAugConsistency solver')

        # This is the dataloader for unsupervised loss
        extra_dataloader = kwargs.get('extra_dataloader')
        self.extra_data_iterator = iter(extra_dataloader)

        self._two_steps = self.config['two_steps_forward_prop']
        self.unsup_loss_weight = self.selection_config['CrossAugConsistency_unsup_loss_weight']
        self.unsup_loss_weight_rampup = self.selection_config['CrossAugConsistency_unsup_loss_weight_rampup']
        self.consistency_num_augmentations = self.selection_config['CrossAugConsistency_num_augmentations']
        self.consistency_alpha_jsd = self.selection_config['CrossAugConsistency_alpha_jsd']
        self.consistency_augmentation_list = eval(self.selection_config['CrossAugConsistency_augmentation_list'])
        self.consistency_aug_gaussian_mean = self.selection_config['CrossAugConsistency_aug_gaussian_mean'] if 'gaussian_noise' in self.consistency_augmentation_list else 0
        self.consistency_aug_gaussian_std = self.selection_config['CrossAugConsistency_aug_gaussian_std'] if 'gaussian_noise' in self.consistency_augmentation_list else 0

        hyper_params = {'param_consistency_loss_weight': self.unsup_loss_weight,
                        'param_consistency_loss_weight_rampup': self.unsup_loss_weight_rampup,
                        'param_two_steps_forward_prop': self._two_steps,
                        'param_consistency_num_augmentations': self.consistency_num_augmentations,
                        'param_consistency_alpha_jsd': self.consistency_alpha_jsd,
                        'param_consistency_augmentation_list': self.consistency_augmentation_list
                        }

        if self.unsup_loss_weight_rampup:
            self.unsup_loss_weight_rampup_length = self.selection_config['CrossAugConsistency_unsup_loss_weight_rampup_length']
            hyper_params['param_consistency_loss_weight_rampup_length'] = self.unsup_loss_weight_rampup_length

        self.experiment.log_parameters(hyper_params)
        
        self.loss_name_list = ['total', 'model', 'consistency']
        
    def train_step(self, data, target):

        # We set all models (task model, single conv's and latentNet) to training mode
        self.models_dic['model'].train()

        # We zero the parameter gradients of all models
        self.optimizers_dic['model'].zero_grad()

        # We put all data unto device
        data, target = data.to(self.device, dtype=torch.float), target.to(self.device)


        # EITHER We do 2 forward propagations through model, once with labeled and one with unlabeled data. Then we forward propagate unlabeled data trhough ema model
        if self._two_steps:
            # We compute the supervised loss
            scores, _ = self.models_dic['model'](data)

            onehot_target = to_onehot(target.squeeze(1), self.models_dic['model'].out_channels)
            model_loss = self.model_loss(scores, onehot_target)

            # We compute the unsupervised loss
            unsup_data, _, _ = next(self.extra_data_iterator)
            unsup_data = unsup_data.to(self.device, dtype=torch.float)

            u_scores, _ = self.models_dic['model'](unsup_data)
            
            # We compute the unsupervised loss
            unsup_loss = self.compute_consistency_loss(u_scores, unsup_data, mode='Train')

            # We combine both to get total loss
            consistency_weight = self.get_current_consistency_weight(self.epoch)
            train_loss = model_loss + consistency_weight * unsup_loss

        # OR We do forward propagation through task model with both labeled and unlabeled data (one forward pass), then we forwrad propagate unlabeled data through ema model
        else:
            # # We get the unsupervised data
            unsup_data, _, _ = next(self.extra_data_iterator)
            unsup_data = unsup_data.to(self.device, dtype=torch.float)

            n_l, n_u = len(data), len(unsup_data)
            
            # We combine labeled and unlabeled data to have only one forward propagation (batch norm reasons)
            all_data = torch.cat([data, unsup_data], dim=0)
            all_scores, _ = self.models_dic['model'](all_data)
            scores, u_scores = torch.split(all_scores, [n_l, n_u], dim=0)

            # We computed the supervised loss
            onehot_target = to_onehot(target, self.models_dic['model'].out_channels)
            model_loss = self.model_loss(scores, onehot_target)

            # We compute the unsupervised loss
            unsup_loss = self.compute_consistency_loss(u_scores, unsup_data, mode='Train')
            #print('unsup_loss {}'.format(unsup_loss))

            # We combine both to get total loss
            consistency_weight = self.get_current_consistency_weight(self.epoch)
            train_loss = model_loss + consistency_weight * unsup_loss

        if self.activate_plot and (self.epoch % self.log_every == 0) and (self.train_batch_idx in [0, 1, 2]):
            self.plot_training_data_pred(data, target, scores, indices=self.idx_l, type='Train')
            self.plot_training_data_pred_contour(data, target, scores, indices=self.idx_l, type='Train')

        # We do backward propagation
        train_loss.backward()
        self.optimizers_dic['model'].step()

        loss_dic = {
            'total': train_loss.item(),
            'model': model_loss.item(),
            'consistency': unsup_loss.item()
        }

        # We compute the metrics
        train_acc_dic = self.compute_metrics(scores, onehot_target)

        return scores, loss_dic, train_acc_dic

    def validation(self, val_dataloader, mode='val'):
        assert mode == 'val' or mode == 'test'
        self.models_dic['model'].eval()

        val_dataloader.dataset.training = False

        # We initialize loss and accuracy
        val_acc_dic = defaultdict(default0)
        val_loss_dic = {}
        for key in self.loss_name_list:
            val_loss_dic[key] = 0

        with torch.no_grad():
            for batch_idx, (val_data, val_target, idx_list) in enumerate(val_dataloader):
                val_data = val_data.to(self.device, dtype=torch.float)
                val_target = val_target.to(self.device)

                val_output, _ = self.models_dic['model'](val_data)
                model_prob = F.softmax(val_output, dim=1)

                onehot_target = to_onehot(val_target.squeeze(1), self.models_dic['model'].out_channels)
                model_loss = self.model_loss(val_output, onehot_target)

                # We compute the unsupervised loss
                unsup_loss = self.compute_consistency_loss(val_output, val_data, mode='Validation')
                
                # We combine both to get total loss
                consistency_weight = self.get_current_consistency_weight(self.epoch)
                total_loss = model_loss + consistency_weight * unsup_loss

                batch_val_loss_dic = {
                    'total': total_loss.item(),
                    'model': model_loss.item(),
                    'consistency': unsup_loss.item()
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

        # We keep track of the consistency weight in comet_ml
        self.experiment.log_metrics({'consistency_weight': consistency_weight}, prefix="consistency_weight", epoch=self.epoch)

        return avg_val_loss_dic, avg_val_acc_dic, True

    def compute_consistency_loss(self, output, data, mode):
        """
        For each sample of the batch, we will do inference on k augmented versions of x' and compute JSD on x' and all T(x')
        """
        # We compte the probability of S(x')
        output_prob = F.softmax(output, dim=1)

        cur_prob_list = [output_prob]
        transformed_data_list = [data]  # REMOVE

        while len(cur_prob_list) < self.consistency_num_augmentations + 1:
            with torch.no_grad():
                # We augment the data
                transformed_data, aug_dic = random_augmentation(
                    data, flip_axis=2, rotaxis0=2, rotaxis1=3, augmentation_list=self.consistency_augmentation_list, type='img',
                    aug_gaussian_mean=self.consistency_aug_gaussian_mean, aug_gaussian_std=self.consistency_aug_gaussian_std)
                transformed_data = transformed_data.to(self.device)
                #print('consistency aug_dic {}'.format(aug_dic))

                # We do inference on the augmented data
                transformed_output, _ = self.models_dic['model'](transformed_data)

                # We do the inverse transformation on the output
                rev_output = reverse_augment_data(transformed_output, aug_dic['flip'], aug_dic['rot'], flip_axis=2, rot_axis0=2, rot_axis1=3)

                # We get output probability and prediction
                rev_prob = F.softmax(rev_output, dim=1)

                # We keep track of the output probabilities
                cur_prob_list.append(rev_prob)
                transformed_data_list.append(transformed_data)  # REMOVE

        # We concatenate the output probability list of augmented inputs
        transformed_prob_concat = torch.stack(cur_prob_list, dim=1)
        
        transformed_data_array = torch.stack(transformed_data_list, dim=1) # REMOVE

        jsd = JSD(transformed_prob_concat, alpha=self.consistency_alpha_jsd, p_ave_dim=1, entropy_aver_dim=1, entropy_dim=2, aver_entropy_dim=1)
        #print('torch.mean(jsd) {}'.format(torch.mean(jsd)))

        if self.activate_plot and (self.epoch % self.log_every == 0) and (self.train_batch_idx in [0, 1, 2]):
            # We plot the predictions
            for i in range(0, 1):
                self.plot_consistency_loss(transformed_data_array[i, :, :, :, :], transformed_prob_concat[i, :, :, :, :], jsd[i, :, :], mode=mode)

        # The tootal unsupervised loss is the average over all data of the batch and all sampel pixels
        unsup_loss = torch.mean(jsd)

        return unsup_loss

    def get_current_consistency_weight(self, epoch):
        """
        # Consistency ramp-up from https://arxiv.org/abs/1610.02242
        Code from https://github.com/HiLab-git/SSL4MIS/blob/10856b2dd7a05a2166744059b958b4915e8a1b5f/code/train_cross_consistency_training_2D.py
        
        The signoid rampup value depends on the current epoch and fixed parameters ( self.unsup_loss_weight and self.unsup_loss_weight_rampup_length)
        """
        return self.unsup_loss_weight * sigmoid_rampup(epoch, self.unsup_loss_weight_rampup_length)

    def plot_consistency_loss(self, aug_data_array, cur_prob_array, jsd, mode):
        fig = plt.figure(figsize=(10, 8))
        nrows, ncols = 2, self.consistency_num_augmentations + 2
        i = 1

        #cur_data = unsup_data[i:i+1, :, :, :][cur_arg, 0, :, :].detach().cpu().numpy()

        # ax = fig.add_subplot(ncols, nrows, i)
        # ax.imshow(cur_data, 'gray', interpolation='none')
        # plt.axis('off')
        # ax.set_title("Data", fontsize=12)
        # i += 1
        i = 1
        for j in range(0, len(aug_data_array)):
            ax = fig.add_subplot(nrows, ncols, i + j)
            cur_data = aug_data_array[j, 0, :, :].detach().cpu().numpy()
            ax.imshow(cur_data, 'gray', interpolation='none')
            plt.axis('off')
            #ax.set_title('{}'.format(flip_rot_pairs[j]), fontsize=12)

        i = ncols
        ax = fig.add_subplot(nrows, ncols, i)
        cur_data = aug_data_array[0,  0, :, :].detach().cpu().numpy()
        ax.imshow(cur_data, 'gray', interpolation='none')
        plt.axis('off')
        ax.imshow(jsd.detach().cpu().numpy(), cmap='viridis', alpha=0.7)
        plt.axis('off')
        ax.set_title('JSD:{:.3f}'.format(torch.mean(jsd)), fontsize=12)

        i = ncols + 1
        for j in range(0, len(cur_prob_array)):
            ax = fig.add_subplot(nrows, ncols, i + j)
            cur_data = aug_data_array[0, 0, :, :].detach().cpu().numpy()
            ax.imshow(cur_data, 'gray', interpolation='none')
            plt.axis('off')
            cur_pred = torch.argmax(cur_prob_array[j, :, :, :], dim=0).detach().cpu().numpy()
            ax.imshow(cur_pred, cmap='viridis', alpha=0.7)
            plt.axis('off')

        # We save the plot
        fig.set_tight_layout({"pad": 0.1})
        savepath = os.path.join('img.png')
        fig.savefig(savepath)
        self.experiment.log_image(savepath, name='{}_consistency_epoch{}'.format(mode, self.epoch), step=self.train_batch_idx)
        plt.close()