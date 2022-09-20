"""
Author: Mélanie Gaillochet
Date: 2020-10-27
"""
import json
import os
import pickle as pkl

import h5py
import nibabel as nib
import numpy as np

from Configs.configs import config_folder


def get_config_from_json(config_filename):
    """
    Get the config from a json file
    :param config_filename: name of config file (located in Configs.config)
    :return: config(namespace) or config(dictionary)
    """
    config_filepath = os.path.join(config_folder, config_filename)
    config_dict = _read_json_file(config_filepath)

    return config_dict


def _read_json_file(file_path):
    """
    We are reading the json file and returning a dictionary
    :param file_path:
    :return:
    """
    # We parse the configurations from the config json file provided
    with open(file_path, 'r') as config_file:
        output_dict = json.load(config_file)
    return output_dict


def load_single_image(folder_path, filename_list, idx):
    """
    We load the data and label for the specific list index given
    :param folder_path:
    :param filename_list:
    :param idx:
    :return: image (array)
    """
    cur_volume_path = os.path.join(folder_path, filename_list[idx])
    ending = cur_volume_path.rpartition('.')[2]

    if ending == 'nii':
        inputImage = nib.load(cur_volume_path)
        img = inputImage.get_data()
        img = np.array(img)

    elif not os.path.isdir(folder_path):
        with h5py.File(folder_path + '.hdf5', 'r') as hf:
            img = hf[filename_list[idx]][:]

    return img


def save_hdf5(data, img_idx, dest_file):
    """
    We are saving an hdf5 object
    :param data:
    :param filename:
    :return:
    """
    with h5py.File(dest_file, "a", libver='latest', swmr=True) as hf:
        hf.swmr_mode = True
        hf.create_dataset(name=str(img_idx), data=data, shape=data.shape, dtype=data.dtype)


def create_unexisting_folder(dir_path):
    """
    We create a folder with the given path.
    If the folder already exists, we add '_1', '_2', ... to it
    :param dir_path:
    """
    i = 0
    created = False
    path = dir_path
    while not created:
        try:
            os.makedirs(path)
            created = True
        except OSError or FileExistsError:
            i += 1
            path = dir_path + '_' + str(i)
            # print(path)

    return path


def save_obj(obj, name):
    """
    Shortcut function to save an object as pkl
    Args:
        obj: object to save
        name: filename of the object
    """
    with open(name + '.pkl', 'wb') as f:
        pkl.dump(obj, f, pkl.HIGHEST_PROTOCOL)


def load_obj(name):
    """
    Shortcut function to load an object from pkl file
    Args:
        name: filename of the object
    Returns:
        obj: object to load
    """
    with open(name + '.pkl', 'rb') as f:
        return pkl.load(f)


def unpickle(file):
    with open(file, 'rb') as fo:
        dict1 = pkl.load(fo, encoding='bytes')
    return dict1
