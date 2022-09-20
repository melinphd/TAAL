"""
Author: Mélanie Gaillochet
Date: 2020-10-05
"""
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

import h5py
import torch
from torch.utils.data import Dataset

from Utils.utils import random_selection
from Utils.augmentation_utils import random_augmentation
from Utils.load_utils import load_single_image
from Utils.utils import natural_keys


class MyDataset(Dataset):
    def __init__(self, data_folder, config):
        self.data_folder = data_folder
        self.dataset_name = config['dataset_name']
        self.num_items = config['num_items']
        try:
            self.downsample_size = eval(config['downsample_size'])
        except TypeError:
            self.downsample_size = None
            
        self.training = False
        self.augment = config['augment']
        if self.augment:
            self.augmentations = eval(config['augmentations'])
            print('\nAugmenting the data with {} \n'.format(self.augmentations))
            self.aug_gaussian_mean = config['aug_gaussian_mean'] if 'gaussian_noise' in  self.augmentations else 0
            self.aug_gaussian_std = config['aug_gaussian_std'] if 'gaussian_noise' in  self.augmentations else 0
            print('self.aug_gaussian_mean {}'.format(self.aug_gaussian_mean))
            print('self.aug_gaussian_std {}'.format(self.aug_gaussian_std))

        # We select the sample paths
        self.volume_folder_path = os.path.join(self.data_folder, self.dataset_name, 'data')
        with h5py.File(self.volume_folder_path + '.hdf5', 'r') as hf:
            self.volume_list = list(hf.keys())
        self.volume_list.sort(key=natural_keys)

        self.seg_folder_path = os.path.join(self.data_folder, self.dataset_name, 'label')
        with h5py.File(self.seg_folder_path + '.hdf5', 'r') as hf:
            self.seg_list = list(hf.keys())
        self.seg_list.sort(key=natural_keys)

        # We select only part of the data
        if self.num_items != "all":
            self.volume_list, self.seg_list = random_selection(self.num_items, self.volume_list,
                                                               self.seg_list)

    def __len__(self):
        """We return the total number of samples"""
        return len(self.volume_list)

    def __getitem__(self, idx):
        """We generate one sample of data"""
        # We load the volume and segmentation samples
        img = load_single_image(self.volume_folder_path, self.volume_list, idx)
        label = load_single_image(self.seg_folder_path, self.seg_list, idx)

        volume = img.copy()
        target = label.copy()

        # We downsample to the given image size
        if self.downsample_size is not None and volume.shape[-1] != self.downsample_size[-1]:
            resized_vol = np.zeros((volume.shape[0],) + self.downsample_size)
            for i in range(0, volume.shape[0]):
                resized_vol[i, :, :] = cv2.resize(np.float32(volume[i, :, :]),
                                                  dsize=self.downsample_size,
                                                  interpolation=cv2.INTER_CUBIC)
            volume = resized_vol.copy()
            target = cv2.resize(np.float32(target),
                                dsize=self.downsample_size,
                                interpolation=cv2.INTER_NEAREST)

        # We convert the volume and segmentation sample to tensors
        volume = torch.from_numpy(volume)
        target = torch.from_numpy(target.astype(float)).long()

        if self.training and self.augment:
            if len(img.shape) == 3:
                flip_axis, rotaxis0, rotaxis1 = 1, 1, 2
            elif len(img.shape) == 4:
                flip_axis, rotaxis0, rotaxis1 = 2, 2, 3
            # We augment with rotations and flips
            volume, aug_dic = random_augmentation(volume, None, flip_axis, rotaxis0, rotaxis1, self.augmentations, 
                                                aug_gaussian_mean=self.aug_gaussian_mean, aug_gaussian_std=self.aug_gaussian_std)
            _target, _ = random_augmentation(target.unsqueeze(0), aug_dic, flip_axis, rotaxis0, rotaxis1, self.augmentations, type='target')
            target = _target[0, :, :]
            
        return volume, target, idx
