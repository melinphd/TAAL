#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Author: Mélanie Gaillochet
Date: 2020-11-18
"""
from comet_ml import Experiment
import argparse
import copy
import json
import math
import os
import random
import time
import pandas as pd
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import torch
import torch.utils.data as data
from torch.utils.data.sampler import SubsetRandomSampler

from Configs.configs import data_folder, output_folder, random_indices_folder
from Data_loader.data_loader import MyDataset
from Enums.model_enum import all_models
from Enums.solver_enum import all_solvers
from Saver.saver import Saver
from Utils.load_utils import get_config_from_json, create_unexisting_folder
from Utils.main_utils import update_kwargs
from Utils.utils import set_all_seed, print_args, convert_time, round_dic_values, NpEncoder
from Utils.sampler_utils import SubsetSequentialSampler, InfiniteSubsetRandomSampler

#from contrastyou.data.creator import get_data


def run_experiment(raw_args=None):
    """
    We run a single experiment with the given configs
    :param raw_args:
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, help='config file path ')
    parser.add_argument('--device', type=str, help='device file path ')
    parser.add_argument('--seed', type=int,
                        help='seed to use for randomness', default=42)
    parser.add_argument('--init_labeled', type=str,
                        help="initial labeled indices if 'labels....' or number of initial indices")
    parser.add_argument('--init_num_labeled', type=int,
                        help="number initial labeled indices")
    args = parser.parse_args(raw_args)
    print_args(args)

    # We load the json config file
    config = get_config_from_json(args.config)
    train_config = config['training']
    batch_size = train_config['batch_size']

    # Making experiment replicable
    set_all_seed(args.seed)
    if train_config['deterministic']:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    iter_start_time = time.time()
    if 'overall_start_time' in config.keys():
        overall_start_time = config['overall_start_time']
    else:
        start_time = datetime.today()
        overall_start_time = '{}_{}h{}min'.format(start_time.date(), start_time.hour,
                                                  start_time.minute)

    print('\nUsing {} sample selection\n'.format(train_config['data_selection']['type']))

    torch.cuda.empty_cache()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # We make the test dataset
    test_data_config = config['data'].copy()
    test_set_name = config['data']['dataset_name'].replace('train', 'test')
    test_data_config['dataset_name'] = test_set_name
    test_data_config['augment'] = False
    test_dataset = MyDataset(data_folder, test_data_config)
    test_dataloader = data.DataLoader(test_dataset, batch_size=1, drop_last=False,
                                        num_workers=train_config['num_workers'],
                                        persistent_workers=True,
                                        pin_memory=True)
    test_dataloader.dataset.training = False

    # We get the test volume lists (for results on 3D volume)
    txt_filepath = os.path.join(data_folder, test_set_name, 'test_indices.txt')
    with open(txt_filepath, encoding='utf8') as f:
        for line in f:
            if 'test_volume_list' in line:
                test_volume_list = line.strip('test_volume_list = ')
                test_volume_list = eval(test_volume_list)

    print('We will be testing on: {} ({} samples from {} volumes)'
            ''.format(test_set_name, len(test_dataset), len(set(test_volume_list))))

    # We load the training data (which allows for augmentations)
    train_dataset = MyDataset(data_folder, config['data'])

    # We load the untransformed dataset (with NO augmentations)
    untrans_data_config = config['data'].copy()
    untrans_data_config['augment'] = False
    dataset = MyDataset(data_folder, untrans_data_config)

    _all_indices = np.arange(len(train_dataset))
    all_indices = _all_indices.tolist()

    # The sample indices used for validation will be the given list or x samples taken randomly
    if isinstance(train_config['val_data'], int):
        val_indices = random.sample(all_indices, train_config['val_data'])
    else:
        val_indices = train_config['val_data']

    # We create a dataloader for validation
    val_sampler = SubsetSequentialSampler(val_indices)
    val_dataloader = data.DataLoader(dataset, sampler=val_sampler, batch_size=batch_size,
                                        drop_last=False, num_workers=train_config['num_workers'],
                                        pin_memory=True)
    val_dataloader.dataset.training = False

    _all_train_indices = np.setdiff1d(all_indices, val_indices)
    all_train_indices = _all_train_indices.tolist()

    # We will take/save the initial indices and subsequent random indices from a given file
    random_indice_filepath = os.path.join(random_indices_folder, 'data_{}_random_init{}_budget{}_seed{}'
                                            ''.format(config['data']['dataset_name'], args.init_labeled,
                                                    train_config['data_selection']['budget'], args.seed))
    print('\nrandom_indice_filepath: {}\n'.format(random_indice_filepath))

    # The labeled sample indices used for training will be the given list or x samples taken randomly
    init_budget = train_config['data_selection']['initial_budget']
    if isinstance(init_budget, list):
        print('\n Taking initial indices from given list')
        _labeled_indices = init_budget
        assert not set(_labeled_indices) & set(val_indices)
    else:
        init_budget = len(
            all_train_indices) if init_budget == "all" else init_budget
        # We check the random indices file to get the initial indices
        try:
            with open(random_indice_filepath + '.txt', encoding='utf8') as f:
                for line in f:
                    if 'init_{}: '.format(init_budget) in line:
                        _indice_list = line.strip('init_{}: '.format(init_budget))
                        _indice_list = _indice_list.strip('\n')
                        _labeled_indices = eval(_indice_list)
                        break
            assert len(_labeled_indices) == init_budget
            print('Initial indices taken from file {}'.format(
                random_indice_filepath + '.txt'))

        # If the initial indice list is not in the file, we will randomly samples them
        except (NameError, FileNotFoundError) as e:
            _labeled_indices = random.sample(all_train_indices, init_budget)

            # If it is the first experiment, then we will save the initial indices
            print('Saving initial indices')
            with open(random_indice_filepath + '.txt', "a") as f:
                f.write('init_{}: {}\n'.format(
                    len(_labeled_indices), _labeled_indices))

    labeled_indices = sorted(_labeled_indices)

    # We initialize the labeled and unlabeled dataloaders
    #sampler = data.sampler.SubsetRandomSampler(labeled_indices, generator=torch.Generator())
    sampler = InfiniteSubsetRandomSampler(
        train_dataset, labeled_indices, shuffle=True)
    querry_dataloader = data.DataLoader(train_dataset, sampler=sampler, batch_size=batch_size,
                                        drop_last=False, num_workers=train_config['num_workers'],
                                        persistent_workers=True,
                                        pin_memory=True)
    querry_dataloader.dataset.training = True

    _unlabeled_indices = np.setdiff1d(all_train_indices, labeled_indices)
    _unlabeled_indices = _unlabeled_indices.tolist()
    unlabeled_indices = sorted(_unlabeled_indices)

    unlabeled_sampler = data.sampler.SubsetRandomSampler(
        unlabeled_indices, generator=torch.Generator())
    #unlabeled_sampler = SubsetSequentialSampler(unlabeled_indices)
    unlabeled_dataloader = data.DataLoader(dataset, sampler=unlabeled_sampler, batch_size=1,
                                            drop_last=False, num_workers=train_config['num_workers'],
                                            persistent_workers=True,
                                            pin_memory=True)
    unlabeled_dataloader.dataset.training = False

    kwargs = {
        'batch_normalization': config['training']['model']['structure']['conv_block']['normalization'],
        'dropout': config['training']['model']['structure']['dropout_rate'],
        'test_volume_list': test_volume_list}

    # For random method or random sampling, we'll take the sampled indices from the file
    if train_config['data_selection']['sampling_type'] == 'Random':
        kwargs['random_indice_filepath'] = random_indice_filepath

    print('\nQuery size: {} (batch size {}), unlabeled size: {}, val size: {}'.format(
        len(labeled_indices), batch_size, len(unlabeled_indices), len(val_indices)))

    # We update the config file
    config['data']['query_size'] = len(labeled_indices)
    config['data']['unlabeled_size'] = len(unlabeled_indices)

    train_config['num_train_iter'] = train_config['num_epochs']

    # We initialize and train all the models
    ModelInit = all_models[train_config['model']['model_name']]
    task_model = ModelInit(train_config['model'])

    # We create a saver to save model, config and losses 
    saver = Saver(config['exp_name'] + 'seed{}_'.format(args.seed) + str(len(labeled_indices)),
                    timestamp=overall_start_time)
    print('Saving model, config and log files in {}'.format(saver.save_folder))
    saver.save_config(config)

    kwargs['model'] = task_model
    kwargs['saver'] = saver
    kwargs['device'] = device
    kwargs['val_dataloader'] = val_dataloader
    kwargs['querry_dataloader'] = querry_dataloader
    kwargs['augment_data'] = config['data']['augment']
    kwargs['augmentations'] = config['data']['augmentations']
    if config['data']['augment'] and 'gaussian_noise' in config['data']['augmentations']:
        kwargs['augmentation_gaussian_mean'] = config['data']['aug_gaussian_mean']
        kwargs['augmentation_gaussian_std'] = config['data']['aug_gaussian_std']
    kwargs['dataset'] = dataset
    kwargs['seed'] = args.seed
    kwargs['labeled_indices'] = labeled_indices

    # We update kwargs according to chosen model
    method = config['training']['data_selection']['type']
    kwargs = update_kwargs(kwargs, method, config, config['training']['data_selection']['sampling_type'], labeled_indices, unlabeled_indices,
                           unlabeled_dataloader, train_dataset,
                           batch_size, args.seed)

    # We initialize the solver
    SolverInit = all_solvers[config['training']['data_selection']['type']]
    solver = SolverInit(config['training'], test_dataloader, **kwargs)

    # We train the models on the current data
    #acc_dic, loss_dic, kwargs_sampler = solver.train()
    acc_dic, loss_dic = solver.train()

    curr_split = len(labeled_indices) / len(all_train_indices)
    nice_last_test_acc_dic = round_dic_values(acc_dic['last_acc'], 4)
    nice_last_3Dtest_acc_dic = round_dic_values(
        acc_dic['3D_last_acc'], 4) if acc_dic['3D_last_acc'] is not None else ''
    print('Final accuracy with {:.3f}% of data (with last model) is: {} and in 3D {}'
          ''.format(curr_split * 100, nice_last_test_acc_dic, nice_last_3Dtest_acc_dic))

    # We sample indices of samples to add to labeled set for the  next experiment
    if len(unlabeled_indices) != 0:
        _sampled_indices = solver.sample_for_labeling(
            unlabeled_dataloader)  # , **kwargs_sampler)
        torch.cuda.empty_cache()
    else:
        _sampled_indices = []
    sampled_indices = sorted(_sampled_indices)

    # We track the runtime
    iter_end_time = time.time()
    min, sec = convert_time(iter_end_time - iter_start_time)
    print('The experiment took {}min {}sec'.format(min, sec))

    summary = 'Split with {:.3f}% of data - query size: {}, unlabeled size: {}, val size: {}' \
              '\n'.format(curr_split * 100, len(labeled_indices), len(unlabeled_indices),
                          len(val_indices))
    summary += 'Best test accuracy: {}  -  best val accuracy: {}\n'.format(acc_dic['best_acc'],
                                                                           acc_dic['val_acc_best'])
    summary += 'Last test accuracy: {}  -  Last val accuracy: {}\n'.format(acc_dic['last_acc'],
                                                                           acc_dic['val_acc_last'])
    summary += 'Best 3D test accuracy: {}\n'.format(
        acc_dic['3D_best_acc'])if acc_dic['3D_best_acc'] is not None else ''
    summary += 'Last 3D test accuracy: {}\n'.format(
        acc_dic['3D_last_acc'])if acc_dic['3D_last_acc'] is not None else ''

    summary += 'Sampling indices: {}\n\n'.format(sampled_indices)

    # (if applicable) We save hyperparameters searched
    hyperparam_dict = config[
        'optimized_hyperparams'] if 'optimized_hyperparams' in config.keys() else ''

    # We save the model, validation score and path in a dictionary
    result_dic = {"num_train_data": len(labeled_indices),
                  "num_val_data": len(val_indices),
                  "test_loss": loss_dic['test_loss'],
                  "test_accuracy_best": acc_dic['best_acc'],
                  "test_accuracy_last": acc_dic['last_acc'],
                  "best_val_loss": loss_dic['val_loss_best'],
                  "best_val_accuracy": acc_dic['val_acc_best'],
                  "last_val_loss": loss_dic['val_loss_last'],
                  "last_val_accuracy": acc_dic['val_acc_last'],
                  "time_taken": '{}min, {}sec'.format(min, sec),
                  "save_path": saver.save_folder,
                  "val_indices": val_indices,
                  "labeled_indices": labeled_indices,
                  "unlabeled_indices": unlabeled_indices,
                  "new_indices_to_sample": sampled_indices,
                  "summary": summary,
                  "hyperparams": hyperparam_dict
                  }

    if acc_dic['3D_last_acc'] is not None:
        result_dic["3Dtest_accuracy_last"] = acc_dic['3D_last_acc']
    if acc_dic['3D_best_acc'] is not None:
        result_dic["3Dtest_accuracy_best"] = acc_dic['3D_best_acc']

    saver.save_txt(result_dic)

    return result_dic


def main(raw_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str,
                        help='path to the config file for the optimization', default='example_config')
    parser.add_argument('--device', type=str,
                        help='device file path ', default='cuda:0')
    parser.add_argument('--seed', type=int,
                        help='seed to use for randomness ', default=42)
    raw_args = parser.parse_args()

    # We load the config file with all parameters to search
    config = get_config_from_json(raw_args.config)

    start_time = datetime.today()

    # We create a folder with txt file and total plots
    log_id = '{}_{}h{}min'.format(
        start_time.date(), start_time.hour, start_time.minute)
    _overall_save_folder = os.path.join(output_folder, log_id, 'overall_results' + '_' +
                                        config['exp_name'] + '_modelLR{}'.format(config['training']['optimizer']['init_lr']) +
                                        '_seed{}'.format(raw_args.seed))
    print(_overall_save_folder)
    overall_save_folder = create_unexisting_folder(_overall_save_folder)
    print('Overall results will be saved in {}'.format(overall_save_folder))

    config['overall_start_time'] = log_id

    # We create a dataframe with table of results
    df_3d_dice = pd.DataFrame(index=['Seed{}_{}'.format(raw_args.seed, raw_args.config)])
    df_2d_dice = pd.DataFrame(index=['Seed{}_{}'.format(raw_args.seed, raw_args.config)])
    df_2d_mIoU = pd.DataFrame(index=['Seed{}_{}'.format(raw_args.seed, raw_args.config)])


    # We will define the number of experiments to run
    # We load the training data
    init_indices = config['training']['data_selection']['initial_budget']
    num_init_indices = init_indices if isinstance(init_indices, int) else len(init_indices)

    _max_iter = math.ceil((config['training']['max_num_labeled'] - num_init_indices) /
                          config['training']['data_selection']['budget'])
    max_iter = int(_max_iter)
    print("\nThere will be {} experiments in total".format(max_iter + 1))

    config_paths = []
    for cur_iter in range(max_iter + 1):
        print('\n ### Experiment {} ###'.format(cur_iter))

        config['training']['cycle'] = str(cur_iter)

        # We update the config file after the first iteration
        updated_config = copy.deepcopy(config)
        if cur_iter > 0:
            print('\n Updating config')
            new_labeled_indices = result_dic['labeled_indices'] + result_dic['new_indices_to_sample']
            updated_config['training']['data_selection']['initial_budget'] = new_labeled_indices
            updated_config['training']['val_data'] = result_dic['val_indices']
        config_path = os.path.join(overall_save_folder, 'config_{}.json'.format(cur_iter))
        config_paths.append(config_path)

        with open(config_paths[-1], 'w') as file:
            json.dump(updated_config, file, indent=4, cls=NpEncoder)

        updated_args = ['--config', config_paths[cur_iter],
                        '--device', raw_args.device,
                        '--seed', str(raw_args.seed),
                        '--init_labeled', str(init_indices).replace('[', 'labels').replace(']', '').replace(',', '-').replace(' ', ''),
                        '--init_num_labeled', str(num_init_indices),
                        ]
        result_dic = run_experiment(updated_args)
        print('\n{}'.format(result_dic['summary']))

        # We add the results to the overall results folder
        txt_save_path = os.path.join(overall_save_folder, 'results.txt')
        with open(txt_save_path, "a") as file_object:
            # Append 'hello' at the end of file
            file_object.write(result_dic['summary'])

        # We put result in the dataframe
        if '3Dtest_accuracy_last' in result_dic.keys():
            # We save 3D dice
            df_3d_dice[result_dic['num_train_data']] = result_dic['3Dtest_accuracy_last']['manual_dice']
            result_name = 'mean_3Dtest_dice_last'
            # We save the transposed results dataframe
            df_3d_dice_t = df_3d_dice.transpose()
            df_3d_dice_savepath = os.path.join(overall_save_folder, result_name + '.csv')
            df_3d_dice_t.to_csv(df_3d_dice_savepath, sep='\t', index=True, header=True)
        
        if 'test_accuracy_last' in result_dic.keys():
            # We save 2D dice
            df_2d_dice[result_dic['num_train_data']] = result_dic['test_accuracy_last']['manual_dice']
            result_name = 'mean_2Dtest_dice_last'
            
            # We save the transposed results dataframe
            df_2d_dice_t = df_2d_dice.transpose()
            df_2d_dice_savepath = os.path.join(overall_save_folder, result_name + '.csv')
            df_2d_dice_t.to_csv(df_2d_dice_savepath, sep='\t', index=True, header=True)
            
            # We save the mean IoU
            df_2d_mIoU[result_dic['num_train_data']] = result_dic['test_accuracy_last']['meanIoU']
            result_name = 'mean_2Dtest_IoU_last'
            
            # We save the transposed results dataframe
            df_2d_mIoU_t = df_2d_mIoU.transpose()
            df_2d_mIoU_savepath = os.path.join(overall_save_folder, result_name + '.csv')
            df_2d_mIoU_t.to_csv(df_2d_mIoU_savepath, sep='\t', index=True, header=True)


    print('Done')


if __name__ == '__main__':
    main()
