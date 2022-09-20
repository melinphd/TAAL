"""
Author: Mélanie Gaillochet
Date: 2020-10-07
"""
import torch
import torch.nn as nn


####### 2D UNet ########
def conv_block_2d(in_channels, out_channels, **kwargs):
    """
    We create a blocks with 3d convolution, batch norm, and ReLU activation
    :param in_channels: number of input channels
    :param out_channels: number of output channels
    """
    normalization = kwargs.get('normalization', None)
    activation_fct = kwargs.get('activation_fct', None)
    kernel_size = kwargs.get('kernel_size', 3)
    stride = kwargs.get('stride', 1)
    padding = kwargs.get('padding', 1)
    momentum = kwargs.get('batch_norm_momentum')

    if normalization == "batch_norm":
        batch_norm = nn.BatchNorm2d(out_channels, momentum=momentum)
    elif normalization == "group_norm":
        batch_norm = nn.GroupNorm(8, out_channels, momentum=momentum)

    # We define the activation function
    if activation_fct == "leakyReLU":
        activation = nn.LeakyReLU(inplace=True)
    elif activation_fct == "ReLU":
        activation = nn.ReLU(inplace=True)

    conv_layer = [
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        batch_norm,
        activation
    ]
    conv_layer = nn.Sequential(*conv_layer)

    return conv_layer


def double_conv_block_2d(in_channels, middle_channels, out_channels, **kwargs):
    """
    We create a layer with 2 convolution blocks (3d conv., BN, and activation)
    :param in_channels: number of input channels
    :param out_channels: number of output channels
    """
    conv_layer = [
        conv_block_2d(in_channels, middle_channels, **kwargs),
        conv_block_2d(middle_channels, out_channels, **kwargs)
    ]
    conv_layer = nn.Sequential(*conv_layer)

    return conv_layer


def max_pooling_2d(**kwargs):
    """
    We apply a max pooling with kernel size 2x2
    :return:
    """
    kernel_size = kwargs.get('kernel_size', 2)
    stride = kwargs.get('stride', 2)
    padding = kwargs.get('padding', 0)

    return nn.MaxPool2d(kernel_size, stride, padding)


def up_conv_2d(in_channels, out_channels, **kwargs):
    """
    We apply and upsampling-convolution with kernel size 2x2
    :param in_channels: # of input channels
    :param out_channels: number of up-convolution channels
    :return:
    """
    kernel_size = kwargs.get('kernel_size', 2)
    stride = kwargs.get('stride', 2)
    padding = kwargs.get('padding', 0)

    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride,
                              padding)


def pad_to_shape(image, shp):
    """
    We pad the input image with zeroes to given shape.
    :param image: the tensor image that we want to pad (has to have dimension 5)
    :param shp: the desired output shape
    :return: zero-padded tensor
    """
    # Pad is a list of length 2 * len(source.shape) stating how many dimensions
    # should be added to the beginning and end of each axis.
    pad = []
    for i in range(len(image.shape) - 1, 1, -1):
        pad.extend([0, shp[i] - image.shape[i]])

    padded_img = torch.nn.functional.pad(image, pad)

    return padded_img
