"""
Author: Mélanie Gaillochet
Date: 2020-11-23
"""
from comet_ml import Experiment

import copy
import os
import time
from collections import defaultdict
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot  as plt
from skimage.measure import find_contours

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.cuda.amp import GradScaler

from Enums.loss_enum import losses
from Enums.metric_enum import metrics
from Utils.optimizer_utils import create_optimizer
from Utils.scheduler_utils import create_scheduler
from Utils.utils import add_dic_values, to_onehot, defaultinf, default0
from Utils.train_utils import compute_metrics, compute_mean_value, print_epoch_update  # , read_data
from Samplers.sampler_random import RandomSampler
from Samplers.sampler_TTA import TestTimeAugmentationSampler
from Samplers.sampler_entropy import PredictionEntropySampler
from Samplers.sampler_coreset import CoresetsSampler
from Samplers.sampler_dropout import DropoutSampler


class BaseSolver:
    """
    This is the base solver class for AL strategies used on the same model
    
    In solver, the following need to be defined:
    - (function) train_step
    - (function) validation
    - (function) sample_for_labeling
    - (variable) self.loss_name_list   # name of model losses
    """

    def __init__(self, config, test_dataloader, **kwargs):
        self.config = config
        self.selection_config = self.config['data_selection']
        self.device = kwargs.get('device')
        self.saver = kwargs.get('saver')

        # We define the data (labeled and val)        
        self.querry_dataloader = kwargs.get('querry_dataloader')
        self.querry_iter = iter(self.querry_dataloader)
        self.val_dataloader = kwargs.get('val_dataloader')
        self.test_dataloader = test_dataloader
        print('Training on batches of size {}, Validating on {} samples, Testing '
              'on {} samples'.format(self.config['batch_size'],
                                     len(self.val_dataloader) * self.val_dataloader.batch_size,
                                     len(self.test_dataloader) * self.test_dataloader.batch_size))
        
        # For test results in 3D
        self.test_volume_list = kwargs.get('test_volume_list')

        # We define task model and its optimizer and scheduler
        self.models_dic = {'model': kwargs.get('model').to(self.device)}
        self.optimizers_dic = {'model': create_optimizer(self.config['optimizer'], self.models_dic['model'].parameters())}
        self.schedulers_dic = {'model': create_scheduler(self.config['scheduler'], self.optimizers_dic['model'])}

        # Training parameters
        self.model_norm_fct = self.config['normalize_fct']
        self.metrics = metrics
        self.main_metric = list(self.metrics.keys())[0]
        self.model_loss = losses[self.config['loss']](normalize_fct=self.config['normalize_fct'])
        self.scale_loss = self.config['scale_loss']
        self.scaler = GradScaler()
        self.sched_early_stop_model_only = self.config['scheduler']['sched_early_stop_model_only']

        # We define number of baches per epoch and number of epochs
        self.batch_size = self.config['batch_size']
        self.num_batches_per_epoch = self.config['num_batches_per_epoch']
        self.epoch = 1
        self.num_epochs = self.config['num_train_iter']
        print('\nThere will be {} training epochs of {} batch iterations'
              '\n'.format(self.num_epochs, self.num_batches_per_epoch))
        self.last_iter = False
        # Here, iters will keep track of every forward-bacward iteration
        self.iters = 0

        # We create a variable that will become positive if the model has converged
        self.early_stop_model = False

        # Whether we will plot validation plots or not and frequency of logs
        self.activate_plot = self.config['activate_plot']
        self.log_every = self.config['log_every']
        print('Will log images every {} epochs'.format(self.log_every))

        # We define sampling strategy and budget
        self.sampling_type = self.selection_config['sampling_type']
        print('\n Sampling with {}'.format(self.sampling_type))
        self.budget = self.selection_config['budget']
        self.labeled_indices = kwargs.get('labeled_indices')
        self.dataset = kwargs.get('dataset')   # To plot the data from its index
        self.sampling_with_best_models = self.selection_config['sampling_with_best_models']
        # If we are using random sampling, we will check the indice file
        self.random_indice_filepath = kwargs.get('random_indice_filepath')

        # Create an experiment with your api key
        self.experiment = Experiment(
            api_key="anonymous",
            project_name="TAAL",
            workspace="anonymous",
        )

        # Report multiple hyperparameters using a dictionary:
        hyper_params = {
            "seed": kwargs.get('seed'),
            "save_file": self.saver.save_folder,
            "selection_type": self.selection_config['type'],
            "deterministic": self.config['deterministic'],
            "data_num_labeled": len(self.selection_config['initial_budget']) if isinstance(self.selection_config['initial_budget'], list) else self.selection_config['initial_budget'],
            "data_num_batches": len(self.querry_dataloader),
            "data_batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
            "augment_data": kwargs.get('augment_data'),
            'augmentation_list': kwargs.get('augmentations') if kwargs.get('augment_data') else [],
            'augmentation_gaussian_mean': kwargs.get('augmentation_gaussian_mean', None),
            'augmentation_gaussian_std': kwargs.get('augmentation_gaussian_std', None),
            "optimizer_model": self.config['optimizer']['optimizer_name'],
            "scheduler_model": self.config['scheduler']['sched_name'],
            "LR_init_model": self.config['optimizer']['init_lr'],
            "model_loss": self.config['loss'],
            "labeled_indices": self.labeled_indices,
            "scale_loss": self.scale_loss,
            "sampling_type": self.sampling_type,
            "sampling_with_best_models": self.sampling_with_best_models,
            "finetune":  kwargs.get('finetune')
        }

        if self.test_volume_list is not None:
            hyper_params["sched_early_stop_model_only"] = self.sched_early_stop_model_only
            hyper_params["model_batch_normalization"] = kwargs.get('batch_normalization')
            hyper_params["model_dropout"] = kwargs.get('dropout')
            hyper_params["test_volume_list"] = self.test_volume_list
            hyper_params["model_finetune_folder"] = kwargs.get('model_finetune_folder')

        # Extra parameters for some sampling strategies
        self.finite_labeled_dataloader = kwargs.get('finite_labeled_dataloader')   # For coresets (finite query dataloader)
        if ('TestTimeAug' in self.sampling_type) or ('Dropout' in self.sampling_type):
            if 'JSD' in self.sampling_type:   # For JSD
                self.alpha_jsd = self.selection_config['alpha_jsd'] 
                print('Setting JSD with alpha {}'.format(self.alpha_jsd))
                hyper_params["param_alpha_jsd"] = self.alpha_jsd
            else:
                self.alpha_jsd = 0.5
        if 'TestTimeAug' in self.sampling_type:
            self.data_aug_gaussian_mean = self.selection_config['data_aug_gaussian_mean']
            self.data_aug_gaussian_std = self.selection_config['data_aug_gaussian_std']
            hyper_params["param_data_aug_gaussian_mean"] = self.data_aug_gaussian_mean
            hyper_params["param_data_aug_gaussian_std"] = self.data_aug_gaussian_std
            print('self.data_aug_gaussian_mean: {}'.format(self.data_aug_gaussian_mean))
            print('self.data_aug_gaussian_std: {}'.format(self.data_aug_gaussian_std))

        self.experiment.log_parameters(hyper_params)

    def train(self):
        # We initialize the default best values
        best_model_val_acc = 0
        best_losses_dic = defaultdict(defaultinf)
        best_models_dic = {}
        last_models_dic = {}
        best_models_dic['model'] = self.models_dic['model']

        # We iterate over all epochs
        while not self.last_iter:
            torch.cuda.empty_cache()
            epoch_start_time = time.time()

            # We train one epoch and do validation
            with self.experiment.train():
                model_train_acc_dic, train_loss_dic, solver_models, lr_dic = self.train_epoch()
                self.experiment.log_metrics(train_loss_dic, prefix="Loss", epoch=self.epoch)
                self.experiment.log_metrics(model_train_acc_dic, prefix="metric", epoch=self.epoch)
                self.experiment.log_metrics(lr_dic, prefix='LR', epoch=self.epoch)
            with self.experiment.validate():
                val_loss_dic, model_val_acc_dic, early_stop_modules = self.validation(
                    self.val_dataloader)
                self.experiment.log_metrics(val_loss_dic, prefix="Loss", epoch=self.epoch)
                self.experiment.log_metrics(model_val_acc_dic, prefix="metric", epoch=self.epoch)

            # We update the model scheduler after each epoch
            self.early_stop_model = self.update_model_scheduler(val_loss_dic['model'])

            # We will save the best validation accuracy
            if model_val_acc_dic[self.main_metric] > best_model_val_acc:
                best_model_val_acc = model_val_acc_dic[self.main_metric]

                if val_loss_dic['model'] < best_losses_dic['model']:
                    best_losses_dic['model'] = val_loss_dic['model']

                    best_models_dic['model'] = copy.deepcopy(solver_models['model']).to('cpu')
                    self.saver.save_best_model(best_models_dic['model'], 'model')

            if self.epoch % self.log_every == 0:
                for cur_model in solver_models.keys():
                    print('Saving {} at epoch {}'.format(cur_model, self.epoch))
                    last_models_dic[cur_model] = copy.deepcopy(solver_models[cur_model]).to('cpu')
                    self.saver.save_model(last_models_dic[cur_model], 'training_' + cur_model + '_epoch' + str(self.epoch))

            # We prepare the variables to be printed
            self.print_epoch_update(train_loss_dic, val_loss_dic, model_train_acc_dic,
                                    model_val_acc_dic,
                                    epoch_start_time,
                                    lr_dic)  

            if self.epoch == self.num_epochs:
                print('Last iteration done')
                self.last_iter = True
            elif self. sched_early_stop_model_only and self.early_stop_model:
                print('Early stopping with model only')
                self.last_iter = True
            elif not self.sched_early_stop_model_only and (self.early_stop_model and early_stop_modules):
                print('Early stopping with model and module')
                self.last_iter = True
            else:
                self.epoch += 1

            if self.last_iter:
                for cur_model in solver_models.keys():
                    last_models_dic[cur_model] = copy.deepcopy(solver_models[cur_model]).to('cpu')
                    self.saver.save_model(last_models_dic[cur_model], 'last_' + cur_model )

        # We evaluate the model on test data
        with self.experiment.test():
            # 2D inference with last model
            last_avg_test_loss, last_test_acc_dic = self.inference(
                last_models_dic['model'].to(self.device),
                self.test_dataloader)
            self.experiment.log_metrics(last_test_acc_dic, prefix='metric_last')

            # 2D inference with best model
            avg_test_loss, test_acc_dic = self.inference(best_models_dic['model'].to(self.device),
                                                         self.test_dataloader)
            self.experiment.log_metrics(test_acc_dic, prefix='metric_best')


            if self.test_volume_list is not None:
                # 3D inference with last model
                _, last_test_acc_dic3d = self.inference3d(last_models_dic['model'].to(self.device),
                                                        self.test_dataloader, self.test_volume_list,
                                                        plot=True)
                self.experiment.log_metrics(last_test_acc_dic3d, prefix='metric3d_last')

                # 3D inference with best model
                _, test_acc_dic3d = self.inference3d(best_models_dic['model'].to(self.device),
                                                    self.test_dataloader, self.test_volume_list,
                                                    plot=False)
                self.experiment.log_metrics(test_acc_dic3d, prefix='metric3d_best')

            else:
                last_test_acc_dic3d = None
                test_acc_dic3d = None

        # We prepare the dictionaries for loss and accuracy results
        acc_dic = {
            "val_acc_last": model_val_acc_dic[self.main_metric],
            "val_acc_best": best_model_val_acc,
            "3D_last_acc": last_test_acc_dic3d,
            "3D_best_acc": test_acc_dic3d,
            "last_acc": last_test_acc_dic,
            "best_acc": test_acc_dic,
        }

        loss_dic = {
            "test_loss": avg_test_loss,
            "val_loss_last": val_loss_dic['total'],
            "val_loss_best": best_losses_dic['total']
        }

        self.experiment.log_parameters({'best_val_loss_model': best_losses_dic['model']})

        return acc_dic, loss_dic#, kwargs_sampler

    def train_epoch(self):
        """
        Training one epoch
        
        We implement the logic of a epoch (before any validation or scheduler update)
        -epoch over the given number of steps
        Returns:
            - avg_model_train_acc_dic
            - train_loss_dic
            - val_loss_dic
            - solver_models
        """
        # We initialize loss and accuracy counts
        model_train_acc_dic = defaultdict(default0)
        train_loss_dic = {}
        for key in self.loss_name_list:
            train_loss_dic[key] = 0

        self.querry_dataloader.dataset.training = True
        for self.train_batch_idx in tqdm(range(self.num_batches_per_epoch)):            
            labeled_imgs, labeled_labels, self.idx_l = next(self.querry_iter)

            # We do a forward-backward prop of the task model and latent model with the labeled data
            _, curr_train_loss_dic, curr_model_train_acc_dic = self.train_step(labeled_imgs, labeled_labels)

            # We add the loss and metrics for each batch
            # Note for accuracy, we put the batch values first because initial model_train_acc_dic is empty
            train_loss_dic = add_dic_values(train_loss_dic, curr_train_loss_dic)
            model_train_acc_dic = add_dic_values(curr_model_train_acc_dic, model_train_acc_dic)
            
            self.iters += 1

        # We compute average accuracy
        avg_train_loss_dic = self.compute_mean_value(train_loss_dic,  self.num_batches_per_epoch)
        avg_model_train_acc_dic = self.compute_mean_value(model_train_acc_dic, self.num_batches_per_epoch)

        # We create an output dictionary of models
        solver_models = {}
        for model_name, cur_model in self.models_dic.items():
            solver_models[model_name] = cur_model

        # We create an output dictionary of LR's
        lr_dic = {}
        for model_name, cur_optim in self.optimizers_dic.items():
            lr_dic[model_name] = cur_optim.param_groups[0]['lr']

        return avg_model_train_acc_dic, avg_train_loss_dic, solver_models, lr_dic

    def train_step(self, data, target):
        """
        We implement the logic of one train step
        Returns:
            - output
            - loss_dic
            - train_acc_dic (any metrics you need to summarize)
        """

        # raise NotImplementedError

    def validation(self, val_dataloader, mode='val'):
        """
        We implement the logic of the validation step.
        Runs once after each epoch
        """
        raise NotImplementedError

    def inference(self, model, test_dataloader):
        """
        We implement the logic of the segmentation inference step.
        Runs once at the end of training
        """
        model.eval()

        test_dataloader.dataset.training = False

        # We initialize loss and accuracy
        loss = 0.0
        acc_dic = defaultdict(default0)

        # We iterate through validation batched
        with torch.no_grad():
            for batch_idx, (data, target, idx_list) in enumerate(test_dataloader):
                data = data.to(self.device, dtype=torch.float)
                target = target.to(self.device)

                output, _ = model(data)

                onehot_target = to_onehot(target, model.out_channels)
                batch_loss = self.model_loss(output, onehot_target)

                # We compute the metrics for the batch
                batch_acc_dic = self.compute_metrics(output, onehot_target)

                # We add the loss and metrics for each batch
                # Note for accuracy, we put the batch values first because initial val_acc_dic is empty
                loss += batch_loss.item()
                acc_dic = add_dic_values(batch_acc_dic, acc_dic)

                # self.save_comet_plot(data, target, output, batch_idx, 'test', img_number=batch_idx)

        # We compute average loss and metric over all batches
        avg_loss = self.compute_mean_value(loss, len(test_dataloader))
        avg_acc_dic = self.compute_mean_value(acc_dic, len(test_dataloader))

        avg_loss_dic = {'model': avg_loss}

        return avg_loss_dic, avg_acc_dic

    def inference3d(self, model, test_dataloader, test_volume_list, plot=False):
        """
        We implement the logic of the segmentation inference step.
        Runs once at the end of training
        """
        model.eval()

        test_dataloader.dataset.training = False

        # We initialize loss and accuracy
        loss = 0.0
        acc_dic = defaultdict(default0)

        data_list = []
        output_list = []
        target_list = []
        onehot_target_list = []

        cur_slice = 0

        # We iterate through validation batched
        with torch.no_grad():
            for batch_idx, (data, target, idx_list) in enumerate(test_dataloader):
                data = data.to(self.device, dtype=torch.float)
                target = target.to(self.device)

                output, _ = model(data)

                onehot_target = to_onehot(target, model.out_channels)

                data_list.append(data)
                output_list.append(output)
                target_list.append(target)
                onehot_target_list.append(onehot_target)

                cur_slice += 1

                # If we reached the last sampled or if we will be changing value in the next samples, we compute the volume metric
                if (batch_idx + 1 == len(test_dataloader)) or (
                        test_volume_list[batch_idx + 1] != test_volume_list[batch_idx]):
                    _vol_data = torch.stack(data_list, dim=0)
                    _vol_output = torch.stack(output_list, dim=0)
                    _vol_target = torch.stack(target_list, dim=0)
                    _vol_onehot_target = torch.stack(onehot_target_list, dim=0)
                    vol_data = _vol_data.permute(1, 2, 0, 3, 4)
                    vol_output = _vol_output.permute(1, 2, 0, 3, 4)
                    vol_target = _vol_target.permute(1, 0, 2, 3)
                    vol_onehot_target = _vol_onehot_target.permute(1, 2, 0, 3, 4)

                    # We compute the metrics for the batch
                    batch_acc_dic = self.compute_metrics(vol_output, vol_onehot_target)
                    batch_loss = self.model_loss(vol_output, vol_onehot_target)

                    # We add the loss and metrics for each batch
                    # Note for accuracy, we put the batch values first because initial val_acc_dic is empty
                    loss += batch_loss.item()
                    acc_dic = add_dic_values(batch_acc_dic, acc_dic)

                    if plot:
                        for slice in range(vol_output.shape[2]):
                            self.save_comet_plot(vol_data[:, :, slice, :, :],
                                                 vol_target[:, slice, :, :],
                                                 vol_output[:, :, slice, :, :], batch_idx, 'test',
                                                 img_number=slice,
                                                 test_name='test_volume_{}'.format(
                                                     test_volume_list[batch_idx]))
                        
                            self.plot_test_data_pred_contour(vol_data[:, :, slice, :, :],
                                                    vol_target[:, slice, :, :],
                                                    vol_output[:, :, slice, :, :], batch_idx, img_idx=slice)

                    data_list = []
                    output_list = []
                    target_list = []
                    onehot_target_list = []
                    cur_slice = 0

        # We compute average loss and metric over all batches
        avg_loss = self.compute_mean_value(loss, len(set(test_volume_list)))
        avg_acc_dic = self.compute_mean_value(acc_dic, len(set(test_volume_list)))

        avg_loss_dic = {'model': avg_loss}

        return avg_loss_dic, avg_acc_dic
    
    def sample_for_labeling(self, unlabeled_dataloader, **kwargs):
        """
        We implement the logic of sample selection for labeling (AL step)
        """
        print('\n sampling_type {}'.format(self.sampling_type))

        unlabeled_dataloader.dataset.training = False

        if self.sampling_type == 'Random':
            try:
                with open(self.random_indice_filepath + '.txt', encoding='utf8') as f:
                    for line in f:
                        if 'sampled_{}: '.format(len(self.labeled_indices)) in line:
                            _indice_list = line.strip(
                                'sampled_{}: '.format(len(self.labeled_indices)))
                            _indice_list = _indice_list.strip('\n')
                            querry_indices = eval(_indice_list)
                assert len(querry_indices) == self.budget
                print('Sampled indices taken from file')
            except (NameError, FileNotFoundError) as e:
                sampler = RandomSampler(self.budget)
                querry_indices = sampler.sample(unlabeled_dataloader)

                with open(self.random_indice_filepath + '.txt', "a") as f:
                    f.write('sampled_{}: {}\n'.format(
                        len(self.labeled_indices), querry_indices))
            uncertainty_values = []

        elif self.sampling_type == 'OutputEntropy':
            sampler = PredictionEntropySampler(self.budget)
            querry_indices, uncertainty_values = sampler.sample(
                self.models_dic['model'], unlabeled_dataloader, self.device, self.experiment)

        elif 'TestTimeAug' in self.sampling_type:
            sampler = TestTimeAugmentationSampler(self.budget)
            querry_indices, uncertainty_values = sampler.sample(
                self.models_dic['model'], unlabeled_dataloader, self.device, self.sampling_type, alpha_jsd=self.alpha_jsd,
                data_aug_gaussian_mean=self.data_aug_gaussian_mean, data_aug_gaussian_std=self.data_aug_gaussian_std)

        elif 'Dropout' in self.sampling_type:
            sampler = DropoutSampler(self.budget)
            querry_indices, uncertainty_values = sampler.sample(
                self.models_dic['model'], unlabeled_dataloader, self.device, self.sampling_type, 
                num_dropout_inference=self.selection_config['dropout_num_inference'], alpha_jsd=self.alpha_jsd)
            
        elif self.sampling_type == 'Coresets':
            pooling_kwargs = {'kernel_size': 4, 'stride': 4, 'padding': 0}

            sampler = CoresetsSampler(self.budget)
            querry_indices = sampler.sample(
                self.models_dic['model'], unlabeled_dataloader, self.device, self.experiment, self.finite_labeled_dataloader, pooling_kwargs)
            uncertainty_values = []
            
        self.experiment.log_other('sampled_indices', querry_indices)
        self.experiment.log_other('uncertainty', uncertainty_values)

        with self.experiment.test():
            self.experiment.log_other('querry_indices', querry_indices)
            for i, sample in enumerate(querry_indices):
                subset = torch.utils.data.Subset(self.dataset, [sample])
                singleloader_subset = torch.utils.data.DataLoader(subset, batch_size=1,
                                                                  num_workers=0, shuffle=False)
                _img, _, _ = next(iter(singleloader_subset))
                img = torch.mean(_img[0, :, :, :].to(torch.float), dim=0)
                self.experiment.log_image(img, name='querry_img', step=sample)
                print(i, sample)
        return querry_indices


    def save_comet_plot(self, val_data, val_target, val_output, batch_idx, mode, idx_list=[],
                        log_val_idx=None, img_number=None, test_name='test_images'):
        # TODO make better
        if len(idx_list) != 0:
            [batch_samples] = np.where(np.array(idx_list) == log_val_idx)
            batch_sample = batch_samples[0]
        else:
            batch_sample = 0
        val_data = val_data.detach().cpu()
        val_target = val_target.detach().cpu()
        val_output = val_output.detach().cpu()

        _prep_val_data = val_data[batch_sample, :, :, :]
        prep_val_data = torch.mean(_prep_val_data, dim=0)

        prep_val_target = val_target[batch_sample, :, :]

        _prep_val_pred = val_output[batch_sample, :, :, :]
        prep_val_pred = torch.argmax(_prep_val_pred, dim=0)

        self.saver.save_pred_img_overlay(prep_val_data, prep_val_target,
                                         prep_val_pred,
                                         filename='epoch{}_batch{}_sample{}'
                                                  ''.format(self.epoch,
                                                            batch_idx,
                                                            batch_sample),
                                         mode=mode)
        img_path = os.path.join(self.saver.save_folder, mode,
                                'epoch{}_batch{}_sample{}_overlay.png'
                                ''.format(self.epoch,
                                          batch_idx,
                                          batch_sample))
        if img_number is None:
            self.experiment.log_image(img_path, name='idx{}_overlay'.format(log_val_idx),
                                      step=self.epoch)
        else:
            self.experiment.log_image(img_path, name=test_name, step=img_number)

    def compute_metrics(self, output, onehot_target):
        return compute_metrics(output, onehot_target, self.metrics, self.model_norm_fct,
                               self.models_dic['model'].out_channels)

    def compute_mean_value(self, input, num_items):
        return compute_mean_value(input, num_items)

    def print_epoch_update(self, train_loss_dic, val_loss_dic, model_train_acc_dic,
                           model_val_acc_dic, epoch_start_time, lr_dic, best_losses_dic='',
                           best_model_val_acc=''):
        return print_epoch_update(self.epoch, train_loss_dic, val_loss_dic, model_train_acc_dic,
                                  model_val_acc_dic, epoch_start_time, lr_dic, best_losses_dic,
                                  best_model_val_acc)

    def update_model_scheduler(self, model_val_loss):
        """
        :param model_val_loss: (optional) needed for a certain type of scheduler
        """
        early_stop = False
        try:
            #self.scheduler_model.step()
            self.schedulers_dic['model'].step()
        except TypeError:
            old_lr = float(self.schedulers_dic['model'].optimizer.param_groups[0]['lr'])
            self.schedulers_dic['model'].step(model_val_loss)
            new_lr = max(old_lr * self.scheduler_model.factor,
                         self.schedulers_dic['model'].min_lrs[0])
            if (old_lr - new_lr < self.schedulers_dic['model'].eps) and \
                    (self.schedulers_dic['model'].num_bad_epochs == self.schedulers_dic['model'].patience):
                early_stop = True
        return early_stop
    
    def plot_training_data_pred(self, data, target, scores, indices, type='Train'):
        """_summary_

        Args:
            data (_type_): _description_
            target (_type_): _description_
            scores (_type_): _description_
            type (str, optional): _description_. Defaults to 'Train'.
            step (_type_, optional): _description_. Defaults to None.
        """
        fig = plt.figure(figsize=(20, 10))
        ncols, nrows = 1, 3

        for img_idx in range(data.shape[0]):
            ## Labeled image
            i = 1
            ax = fig.add_subplot(ncols, nrows, i)
            ax.imshow(data[img_idx, 0, :, :].detach().cpu().numpy(), 'gray')
            plt.axis('off')
            ax.set_title("Data", fontsize=12)

            i = 2
            ax = fig.add_subplot(ncols, nrows, i)
            ax.imshow(data[img_idx, 0, :, :].detach().cpu().numpy(), 'gray', interpolation='none')
            plt.axis('off')
            ax.imshow(target.squeeze(1)[img_idx, :, :].detach().cpu().numpy(), cmap='viridis', alpha=0.7)
            plt.axis('off')
            ax.set_title("Target", fontsize=12)
    
            i = 3
            ax = fig.add_subplot(ncols, nrows, i)
            ax.imshow(data[img_idx, 0, :, :].detach().cpu().numpy(), 'gray', interpolation='none')
            plt.axis('off')
            model_pred = torch.argmax(scores, dim=1)
            ax.imshow(model_pred[img_idx, :, :].detach().cpu().numpy(), cmap='viridis', alpha=0.7)
            plt.axis('off')
            ax.set_title("Pred Model", fontsize=12)

            # We save the plot
            fig.set_tight_layout({"pad": 0.})
            savepath = os.path.join('img.png')
            fig.savefig(savepath)
            self.experiment.log_image(savepath, name='{}_predictions_epoch{}'.format(type, self.epoch), step=indices[img_idx])
            plt.close()

    def plot_training_data_pred_contour(self, data, target, scores, indices, type='Train'):
        """_summary_

        Args:
            data (_type_): _description_
            target (_type_): _description_
            scores (_type_): _description_
            type (str, optional): _description_. Defaults to 'Train'.
            step (_type_, optional): _description_. Defaults to None.
        """
        fig = plt.figure(figsize=(10, 10))
        ncols, nrows = 1, scores.shape[1] - 1

        for img_idx in range(data.shape[0]):

            cur_data = data[img_idx, 0, :, :].detach().cpu().numpy()
            cur_target = target.squeeze(1)[img_idx, :, :].detach().cpu().numpy()
            cur_pred = torch.argmax(scores, dim=1)[img_idx, :, :].detach().cpu().numpy()

            # Computing the Active Contour for the given image
            contour_target = find_contours(cur_target.T, 0.5)
            contour_pred = find_contours(cur_pred.T, 0.5)

            ## Labeled image
            ax = fig.add_subplot(ncols, nrows, 1)
            ax.imshow(cur_data, 'gray')
            plt.axis('off')
            for contour in contour_target:
                ax.plot(contour[:, 0], contour[:, 1], '-b', lw=5)
                plt.axis('off')
            for contour in contour_pred:
                ax.plot(contour[:, 0], contour[:, 1], '-r', lw=5)
                plt.axis('off')

            # We save the plot
            fig.set_tight_layout({"pad": 0.})
            savepath = os.path.join('img.png')
            fig.savefig(savepath)
            self.experiment.log_image(savepath, name='{}_Predictions_epoch{}_Contour'.format(type, self.epoch), step=indices[img_idx])
            plt.close()

    def plot_val_data_pred(self, val_data, val_target, val_output, batch_idx, img_idx=0):
        fig = plt.figure(figsize=(20, 10))
        ncols, nrows = 1, 3

        ## Labeled image
        i = 1
        ax = fig.add_subplot(ncols, nrows, i)
        ax.imshow(val_data[img_idx, 0, :,:].detach().cpu().numpy(), 'gray')
        plt.axis('off')
        ax.set_title("Data", fontsize=12)

        i = 2
        ax = fig.add_subplot(ncols, nrows, i)
        ax.imshow(val_data[img_idx, 0, :, :].detach().cpu().numpy(), 'gray', interpolation='none')
        plt.axis('off')
        ax.imshow(val_target[img_idx, :, :].detach().cpu().numpy(), cmap='viridis', alpha=0.7)
        ax.imshow(val_target.squeeze(1)[img_idx, :, :].detach().cpu().numpy(), cmap='viridis', alpha=0.7)
        plt.axis('off')
        ax.set_title("Target", fontsize=12)

        i = 3
        ax = fig.add_subplot(ncols, nrows, i)
        ax.imshow(val_data[img_idx, 0, :, :].detach().cpu().numpy(), 'gray', interpolation='none')
        plt.axis('off')
        val_pred = torch.argmax(val_output, dim=1)
        ax.imshow(val_pred[img_idx, :, :].detach().cpu().numpy(), cmap='viridis', alpha=0.7)
        plt.axis('off')
        ax.set_title("Pred", fontsize=12)

        # We save the plot
        fig.set_tight_layout({"pad": 0.})
        savepath = os.path.join('img.png')
        fig.savefig(savepath)
        self.experiment.log_image(savepath, name='Validation_predictions_batch{}_img{}'.format(batch_idx, img_idx), step=self.epoch)
        plt.close()

    def plot_val_data_pred_contour(self, val_data, val_target, val_output, batch_idx, img_idx=0):
        fig = plt.figure(figsize=(10, 10))
        ncols, nrows = 1, val_output.shape[1] - 1

        cur_data = val_data[img_idx, 0, :, :].detach().cpu().numpy()
        cur_target = val_target.squeeze(1)[img_idx, :, :].detach().cpu().numpy()
        cur_pred = torch.argmax(val_output, dim=1)[img_idx, :, :].detach().cpu().numpy()

        # Computing the Active Contour for the given image
        contour_target = find_contours(cur_target.T, 0.5)
        contour_pred = find_contours(cur_pred.T, 0.5)

        ## Labeled image
        ax = fig.add_subplot(ncols, nrows, 1)
        ax.imshow(cur_data, 'gray')
        plt.axis('off')
        for contour in contour_target:
            ax.plot(contour[:, 0], contour[:, 1], '-b', lw=5)
            plt.axis('off')
        for contour in contour_pred:
            ax.plot(contour[:, 0], contour[:, 1], '-r', lw=5)
            plt.axis('off')

        # We save the plot
        fig.set_tight_layout({"pad": 0.})
        savepath = os.path.join('img.png')
        fig.savefig(savepath)
        self.experiment.log_image(savepath, name='Validation_predictions_batch{}_img{}_Contour'.format(
            batch_idx, img_idx), step=self.epoch)
        plt.close()

    def plot_test_data_pred_contour(self, val_data, val_target, val_output, batch_idx, img_idx=0):
        fig = plt.figure(figsize=(10, 10))
        ncols, nrows = 1, val_output.shape[1] - 1

        cur_data = val_data[0, 0, :, :].detach().cpu().numpy()
        cur_target = val_target.squeeze(1)[0, :, :].detach().cpu().numpy()
        cur_pred = torch.argmax(val_output, dim=1)[0, :, :].detach().cpu().numpy()

        # Computing the Active Contour for the given image
        contour_target = find_contours(cur_target.T, 0.5)
        contour_pred = find_contours(cur_pred.T, 0.5)

        ## Labeled image
        ax = fig.add_subplot(ncols, nrows, 1)
        ax.imshow(cur_data, 'gray')
        plt.axis('off')
        for contour in contour_target:
            ax.plot(contour[:, 0], contour[:, 1], '-b', lw=5)
            plt.axis('off')
        for contour in contour_pred:
            ax.plot(contour[:, 0], contour[:, 1], '-r', lw=5)
            plt.axis('off')

        # We save the plot
        fig.set_tight_layout({"pad": 0.})
        savepath = os.path.join('img.png')
        fig.savefig(savepath)
        self.experiment.log_image(savepath, name='Test_predictions_vol{}_Contour'.format(batch_idx), step=img_idx)
        plt.close()
