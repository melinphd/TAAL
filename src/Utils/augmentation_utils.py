"""
Author: Mélanie Gaillochet
Date: 2021-02-22
"""
from comet_ml import Experiment
import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from Utils.loss import DiceLoss
from Utils.utils import normalize
from Utils.train_utils import softmax_with_temp


# Loss function and optimizer
def entropy(p):
    """ We compute the entropy """
    if (len(p.size())) == 2:
        return - torch.sum(p * torch.log(p + 1e-18)) / float(len(p))
    elif (len(p.size())) == 1:
        return - torch.sum(p * torch.log(p + 1e-18))
    else:
        raise NotImplementedError


def Compute_entropy(net, x):
    """ We compute the conditional entropy H(Y|X) and the entropy H(Y) """
    p = F.softmax(net(x), dim=1)
    p_ave = torch.sum(p, dim=0) / len(x)
    return entropy(p), entropy(p_ave)


def Compute_entropies_1d(output):
    """ We compute the conditional entropy H(Y|X) and the entropy H(Y) """
    p = F.softmax(output, dim=1)
    aver_entropy = entropy(p)
    p_ave = torch.sum(p, dim=0) / len(output)
    entropy_aver = entropy(p_ave)
    return aver_entropy, entropy_aver


def entropy_2d(p, dim=0, keepdim=False):
    """ 
    We compute the entropy along the first dimension, for each value of the tensor
    :param p: tensor of probabilities
    :param dim: dimension along which we want to compute entropy (the sum across this dimension must be equal to 1)
    """
    entrop = - torch.sum(p * torch.log(p + 1e-18), dim=dim, keepdim=keepdim)
    return entrop


def compute_aver_entropy_2d(prob_input, entropy_dim=1, aver_entropy_dim=[1, 2]):
    tot_entropy = entropy_2d(prob_input, dim=entropy_dim)
    aver_entropy = torch.mean(tot_entropy, dim=aver_entropy_dim)
    return aver_entropy


def compute_entropy_aver_2d(prob_input, p_ave_dim=[2, 3], entropy_aver_dim=1):
    p_ave = torch.mean(prob_input, dim=p_ave_dim)
    entropy_aver = entropy_2d(p_ave, dim=entropy_aver_dim)
    return entropy_aver


def Compute_entropies_2d(output, softmax_temp=None):
    """ We compute the conditional entropy H(Y|X) and the entropy H(Y) """
    if softmax_temp is None:
        p = F.softmax(output, dim=1)
    else:
        output_with_temp = torch.div(output, softmax_temp)
        p = F.softmax(output_with_temp, dim=1)
    aver_entropy = compute_aver_entropy_2d(p)
    entropy_aver = compute_entropy_aver_2d(p)
    return aver_entropy, entropy_aver


def kl(p, q):
    """ We compute KL divergence between p and q """
    return torch.sum(p * torch.log((p + 1e-8) / (q + 1e-8))) / float(len(p))


def distance(y0, y1):
    # compute KL divergence between the outputs of the network
    return kl(F.softmax(y0, dim=1), F.softmax(y1, dim=1))


def JSD(prob_dists, alpha, p_ave_dim=0, entropy_aver_dim=0, entropy_dim=1, aver_entropy_dim=0):
    """
    JS divergence JSD(p1, .., pn) = H(sum_i_to_n [w_i * p_i]) - sum_i_to_n [w_i * H(p_i)], where w_i is the weight given to each probability
                                  = Entropy of average prob. - Average of entropy

    :param prob_dists: probability tensors (shape #points-to-compare, #channels, H, W)
    :param alpha: weight on terms of the JSD
    """
    entropy_mean = compute_entropy_aver_2d(prob_dists, p_ave_dim=p_ave_dim, entropy_aver_dim=entropy_aver_dim)
    mean_entropy = compute_aver_entropy_2d(prob_dists, entropy_dim=entropy_dim, aver_entropy_dim=aver_entropy_dim)
    jsd = alpha * entropy_mean - (1 - alpha) * mean_entropy
    
    return jsd


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        tensor = tensor.detach().cpu()
        noise = torch.randn(tensor.size())
        return tensor + noise * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


def random_augmentation(x, aug_dic=None, flip_axis=2, rotaxis0=2, rotaxis1=3, augmentation_list=[], type='img',  aug_gaussian_mean=0,  aug_gaussian_std=0):
    """ We do augmentation (flip, rotation, mult(0.9 - 1.1)
    :param x: a tensor of shape (#channels, x, y) or (#channels, x, y, z)
    :param aug_dic: augmentation dictionary (if given)
    :param flip_axis: tensor axis for flipping
    :param rotaxis0: tensor first axis of rotation
    :param rotaxis1: tensor second axis of rotation
    :param type: type of input ('img' or 'target'). If 'target', no jitter or blurring will be applied
    """
    if aug_dic is None:
        # We get params for number of flips (0 or 1) and number of rotations (0 ro 3)
        flip = torch.randint(0, 2, (1,)).item() if 'flip' in augmentation_list else 0
        num_rot = torch.randint(0, 4, (1,)).item() if 'rotation' in augmentation_list else 0
        
        # We define the same value for amount of brightness, contrast, saturation and hue jitter.
        # The factor will be uniformly from [max(0, 1 - value), 1 + value], 
        # except for hue value which will be chosen between  0.5 < =[-value, value] <= 0.5
        jitter = 0.5 if 'jitter' in augmentation_list else 0
        
        # We define the same value for kernel size and max sigma. 
        # Sigma will be chosen uniformly at random between (0.1, value)
        blur = 3 if 'blur' in augmentation_list else 1
        
        mean_gaussian = aug_gaussian_mean if 'gaussian_noise' in augmentation_list else 0
        std_gaussian = aug_gaussian_std if 'gaussian_noise' in augmentation_list else 0

        aug_dic = {'flip': flip,
                    'rot': num_rot,
                    'jitter': jitter,
                    'blur': blur,
                    'mean_gaussian': mean_gaussian,
                    'std_gaussian': std_gaussian
                    }
    else:
        flip = aug_dic['flip']
        num_rot = aug_dic['rot']
        
        # If it is a target image, there will be no jitter and bluring transformation
        jitter = 0 if type == 'target' else aug_dic['jitter']
        blur = 1 if type == 'target' else aug_dic['blur']
        mean_gaussian = 0 if type == 'target' else aug_dic['mean_gaussian']
        std_gaussian = 0  if type == 'target' else aug_dic['std_gaussian']

    # We apply the transformations
    x_aug = augment_data(x, flip=flip, n_rotation=num_rot, flip_axis=flip_axis, rot_axis0=rotaxis0, rot_axis1=rotaxis1,
                         jitter=jitter, blur=blur, mean_gaussian=mean_gaussian, std_gaussian=std_gaussian)

    return x_aug, aug_dic


def augment_data(img, flip=0, n_rotation=0, flip_axis=2, rot_axis0=2, rot_axis1=3, jitter=0, blur=1, mean_gaussian=0, std_gaussian=0):
    """
    We apply the given transformation (flip and rotation) on the input image
    :param flip: [0 or 1] flip applied as the initial transformation
    :param flip: [0, 1, 2, 3] number of rotations applied as the initial transformation
    :param jitter:  (same) value for amount of brightness, contrast, saturation and hue jitter.
                    The factor will be uniformly from [max(0, 1 - value), 1 + value], 
                    except for hue value which will be chosen between  0.5 < =[-value, value] <= 0.5
    :param blur: (same) value of kernel size and sigma for Gaussian blur. Kernel will have shape (value, value)
                 Sigma will be chosen uniformly at random between 0.1 and that value.
    """
    if flip != 0:
        img = torch.flip(img, [flip_axis])
        
    if n_rotation !=0:
        img = torch.rot90(img, n_rotation, [rot_axis0, rot_axis1])
    
    if jitter != 0:
        transform = T.ColorJitter(brightness=jitter, contrast=jitter, saturation=jitter, hue=jitter)
        img = transform(img)
        
    if blur != 1:
        transform = T.GaussianBlur(kernel_size=(blur,blur), sigma=(0.1, blur))
        img = transform(img)
        
    if mean_gaussian != 0 or std_gaussian != 0:
        transform = AddGaussianNoise(mean_gaussian, std_gaussian)
        img = transform(img)
        
    return img

def reverse_augment_data(img, flip=0, n_rotation=0, flip_axis=2, rot_axis0=2, rot_axis1=3):
    """
    We reverse the transformation (flip and rotation) of the given image
    :param flip: [0 or 1] flip applied as the initial transformation
    :param flip: [0, 1, 2, 3] number of rotations applied as the initial transformation
    """
    if n_rotation !=0:
        img = torch.rot90(img, 4 - n_rotation, [rot_axis0, rot_axis1])
        
    if flip != 0:
        img = torch.flip(img, [flip_axis])
        
        
    return img


def compute_aug_loss(aug_loss_type, unsup_trans_output, unsup_output_aug, detach_trans,
                     model_norm_fct):
    if aug_loss_type == 'consistency_regularization_pixelKL':
        log_softmax = torch.nn.LogSoftmax(dim=1)
        log_softmax_output_aug = log_softmax(unsup_output_aug)
        log_softmax_trans_output = log_softmax(unsup_trans_output)

        KL_loss = torch.nn.KLDivLoss(reduction='mean', log_target=True)
        aug_loss = KL_loss(log_softmax_trans_output.detach(),
                           log_softmax_output_aug) if detach_trans else KL_loss(
            log_softmax_trans_output, log_softmax_output_aug)

    elif aug_loss_type == 'consistency_regularization_symmetricpixelKL':
        log_softmax = torch.nn.LogSoftmax(dim=1)
        log_softmax_output_aug = log_softmax(unsup_output_aug)
        log_softmax_trans_output = log_softmax(unsup_trans_output)

        KL_loss = torch.nn.KLDivLoss(reduction='mean', log_target=True)
        aug_loss = 0.5 * KL_loss(log_softmax_trans_output.detach(), log_softmax_output_aug) + \
                   0.5 * KL_loss(log_softmax_output_aug,
                                 log_softmax_trans_output.detach()) if detach_trans \
            else 0.5 * KL_loss(log_softmax_trans_output, log_softmax_output_aug) + \
                 0.5 * KL_loss(log_softmax_output_aug, log_softmax_trans_output)

    elif aug_loss_type == 'consistency_regularization_dice':
        dice = DiceLoss(normalize_fct=model_norm_fct)
        aug_loss = dice(unsup_trans_output.detach(), normalize(model_norm_fct, unsup_output_aug)) \
            if detach_trans \
            else dice(unsup_trans_output, normalize(model_norm_fct, unsup_output_aug))

    elif aug_loss_type == 'consistency_regularization_L2':
        mse = torch.nn.MSELoss()
        aug_loss = mse(F.softmax(unsup_trans_output.detach(), dim=1),
                       F.softmax(unsup_output_aug, dim=1)) \
            if detach_trans \
            else mse(F.softmax(unsup_trans_output, dim=1), F.softmax(unsup_output_aug, dim=1))

    return aug_loss


def log_aug_train_images(unsup_data_aug, unsup_trans_output, unsup_output_aug, saver, experiment,
                         epoch, train_batch_idx):
    batch_sample = 0
    _prep_aug_data = unsup_data_aug.detach().cpu()[batch_sample, :, :, :]
    prep_aug_data = torch.mean(_prep_aug_data, dim=0)

    _prep_trans_output = unsup_trans_output.detach().cpu()[batch_sample, :, :, :]
    prep_trans_output = torch.argmax(_prep_trans_output, dim=0)

    _prep_aug_output = unsup_output_aug.detach().cpu()[batch_sample, :, :, :]
    prep_aug_output = torch.argmax(_prep_aug_output, dim=0)
    saver.save_pred_img_overlay(prep_aug_data, prep_trans_output, prep_aug_output,
                                filename='epoch{}_batch{}_sample{}'
                                         ''.format(epoch, train_batch_idx,
                                                   batch_sample),
                                mode='train')
    img_path = os.path.join(saver.save_folder, 'train',
                            'epoch{}_batch{}_sample{}_overlay.png'
                            ''.format(epoch, train_batch_idx,
                                      batch_sample))
    experiment.log_image(img_path,
                         name='train_aug_overlay',
                         step=epoch)
