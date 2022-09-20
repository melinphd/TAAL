"""
Author: Mélanie Gaillochet
Date: 2020-10-06
"""
import json
import os
from datetime import datetime

import SimpleITK as sitk
import matplotlib
import numpy as np
import pandas as pd
# import seaborn as sns
import torch
from skimage.util import montage as montage2d
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# from torch.utils.tensorboard import SummaryWriter

matplotlib.rc('xtick', labelsize=22)
matplotlib.rc('ytick', labelsize=22)
import matplotlib.pyplot as plt
from skimage import measure

plt.rcParams.update({'font.size': 28})

from Configs.configs import output_folder
from Utils.load_utils import create_unexisting_folder, save_obj


class Saver(object):
    def __init__(self, exp_name, train=True, timestamp=None):

        if train:
            # We create a folder to save the results
            self.save_folder = self._create_run_dir(exp_name, timestamp)
            self.save_training_path = os.path.join(self.save_folder,
                                                   'training_state.pth.tar')

            # # We will save entries in the log_dir to be shown in TensorBoard
            # self.writer = SummaryWriter(self.save_folder)
        else:
            test_folder = os.path.join(output_folder, 'inference', exp_name)
            self.save_folder = create_unexisting_folder(test_folder)

    def _create_run_dir(self, exp_name, dtime=None):
        """
        We create the output folder of the experiment, if it doesn't exist yet
        """
        dtime = datetime.today() if dtime is None else dtime
        if not isinstance(dtime, str):
            log_id = '{}_{}h{}min'.format(dtime.date(), dtime.hour,
                                          dtime.minute)
        else:
            log_id = dtime
        save_folder = os.path.join(output_folder, log_id, exp_name)
        save_folder = create_unexisting_folder(save_folder)
        return save_folder

    def save_model(self, model, model_name):
        save_path = os.path.join(self.save_folder, model_name + '.pt')
        torch.save(model.state_dict(), save_path)

    def save_best_model(self, model, model_name):
        save_best_model_path = os.path.join(self.save_folder,
                                            'best_{}.pt'.format(model_name))
        torch.save(model.state_dict(), save_best_model_path)

    def save_training_state(self, cur_iter, models_dic, scheduler, optimizer,
                            model_val_loss_list, cur_metric_dic, best_loss, best_metric):
        """
        We save the training state for inference or to continue training
        :param cur_epoch:
        :param model:
        :return:
        """
        training_state = {'iter': cur_iter,
                          'lr_scheduler': scheduler.state_dict(),
                          'optimizer': optimizer.state_dict(),
                          'model_loss_list': model_val_loss_list,
                          'cur_metric': cur_metric_dic,
                          'best_loss': best_loss,
                          'best_metric': best_metric,
                          }
        for model_name, model_value in models_dic.items():
            training_state[
                'state_dict_{}'.format(model_name)] = model_value.state_dict(),

        torch.save(training_state, self.save_training_path)

    def save_config(self, config):
        """
        We are saving the given config file
        """
        with open(os.path.join(self.save_folder, 'config.json'), 'w') as file:
            json.dump(config, file, indent=4)

    def save_txt(self, dic):
        """
        We are saving the input dictionary as a test file
        """
        txt_save_path = os.path.join(self.save_folder, 'run_info.txt')
        with open(txt_save_path, "w") as file:
            for key in dic.keys():
                file.write('{} = {}\n'.format(key, dic[key]))
            file.write('\n')
