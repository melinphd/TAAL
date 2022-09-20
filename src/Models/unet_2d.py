"""
Author: Mélanie Gaillochet
Date: 2021-02-12
"""
import torch
import torch.nn as nn

from Utils.unet_utils import double_conv_block_2d, max_pooling_2d, up_conv_2d, \
    pad_to_shape


class ModUNet2D(nn.Module):
    def __init__(self, config):
        """
        This modified UNet differs from the original one with the use of
        leaky relu (instead of ReLU) and the addition of residual connections.
        The idea is to help deal with fine-grained details
        :param in_channels: # of input channels (ie: 3 if  image in RGB)
        :param out_channels: # of output channels (# segmentation classes)
        """
        super(ModUNet2D, self).__init__()

        self.in_channels = config["in_channels"]
        self.out_channels = config["out_channels"]
        self.num_init_filters = config["num_init_filters"]

        # Kernel size, stride, paddding, normalizatin and activation function will be passed as keyword arguments
        kwargs = config['structure']

        # We set the dropout
        self.dropout = nn.Dropout(kwargs['dropout_rate'])

        # Encoder part
        self.enc_1 = double_conv_block_2d(self.in_channels,
                                          self.num_init_filters // 2,
                                          self.num_init_filters,
                                          **kwargs['conv_block'])
        self.pool_1 = max_pooling_2d(**kwargs['pooling'])

        self.enc_2 = double_conv_block_2d(self.num_init_filters,
                                          self.num_init_filters,
                                          self.num_init_filters * 2,
                                          **kwargs['conv_block'])
        self.pool_2 = max_pooling_2d(**kwargs['pooling'])
        self.enc_3 = double_conv_block_2d(self.num_init_filters * 2,
                                          self.num_init_filters * 2,
                                          self.num_init_filters * 4,
                                          **kwargs['conv_block'])
        self.pool_3 = max_pooling_2d(**kwargs['pooling'])

        # Center part
        self.center = double_conv_block_2d(self.num_init_filters * 4,
                                           self.num_init_filters * 4,
                                           self.num_init_filters * 8,
                                           **kwargs['conv_block'])

        # Decoder part
        self.up_1 = up_conv_2d(self.num_init_filters * 8,
                               self.num_init_filters * 8, **kwargs['upconv'])
        self.dec_1 = double_conv_block_2d(self.num_init_filters * 12,
                                          self.num_init_filters * 4,
                                          self.num_init_filters * 4,
                                          **kwargs['conv_block'])
        self.up_2 = up_conv_2d(self.num_init_filters * 4,
                               self.num_init_filters * 4, **kwargs['upconv'])
        self.dec_2 = double_conv_block_2d(self.num_init_filters * 6,
                                          self.num_init_filters * 2,
                                          self.num_init_filters * 2,
                                          **kwargs['conv_block'])
        self.up_3 = up_conv_2d(self.num_init_filters * 2,
                               self.num_init_filters * 2, **kwargs['upconv'])
        self.dec_3 = double_conv_block_2d(self.num_init_filters * 3,
                                          self.num_init_filters,
                                          self.num_init_filters,
                                          **kwargs['conv_block'])

        # Output
        self.out = nn.Conv2d(self.num_init_filters, self.out_channels,
                             kernel_size=1, padding=0)

    def forward(self, x):
        # Encoding
        # print('\nEncoder 1')
        enc_1 = self.enc_1(x)  # -> [BS, 64, x, y, z], if num_init_filters=64
        # print(enc_1.shape)
        out = self.pool_1(enc_1)  # -> [BS, 64, x/2, y/2, z/2]

        # We put a dropout layer
        out = self.dropout(out)

        # print('\nEncoder 2')
        enc_2 = self.enc_2(out)  # -> [BS, 128, x/2, y/2, z/2]
        # print(enc_2.shape)
        out = self.pool_2(enc_2)  # -> [BS, 128, x/4, y/4, z/4]

        # We put a dropout layer
        out = self.dropout(out)

        # print('\nEncoder 3')
        enc_3 = self.enc_3(out)  # -> [BS, 256, x/4, y/4, z/4]
        # print(enc_3.shape)
        out = self.pool_3(enc_3)  # -> [BS, 256, x/8, y/8, z/8]

        # We put a dropout layer
        out = self.dropout(out)

        # Center
        # print('\nCenter')
        center = self.center(out)  # -> [BS, 512, x/8, y/8, z/8]
        # print(center.shape)

        # Decoding
        # print('\nDecoder 1')
        out = self.up_1(center)  # -> [BS, 512, x/4, y/4, z/4]
        # print(out.shape)
        out = pad_to_shape(out, enc_3.shape)
        # print(out.shape)
        out = torch.cat([out, enc_3], dim=1)  # -> [BS, 768, x/4, y/4, z/4]
        # print(out.shape)
        out = self.dropout(out)
        dec_1 = self.dec_1(out)  # -> [BS, 256, x/4, y/4, z/4]
        # print(dec_1.shape)

        # print('\nDecoder 2')
        out = self.up_2(dec_1)  # -> [BS, 256, x/2, y/2, z/2]
        # print(out.shape)
        out = pad_to_shape(out, enc_2.shape)
        # print(out.shape)
        out = torch.cat([out, enc_2], dim=1)  # -> [BS, 384, x/2, y/2, z/2]
        # print(out.shape)
        out = self.dropout(out)
        dec_2 = self.dec_2(out)  # -> [BS, 128, x/2, y/2, z/2]
        # print(dec_2.shape)

        # print('\nDecoder 3')
        out = self.up_3(dec_2)  # -> [BS, 128, x, y, z]
        # print(out.shape)
        out = pad_to_shape(out, enc_1.shape)
        # print(out.shape)
        out = torch.cat([out, enc_1], dim=1)  # -> [BS, 192, x, y, z]
        # print(out.shape)
        out = self.dropout(out)
        dec_3 = self.dec_3(out)  # -> [BS, 64, x, y, z]
        # print(dec_3.shape)

        # We put a dropout layer
        # out = self.dropout(dec_3)

        # Output
        # print('\nOutput')
        out = self.out(dec_3)  # -> [BS, out_channels, x, y, z]
        # print(out.shape)
        return out, [enc_1, enc_2, enc_3, center, dec_1, dec_2, dec_3]


if __name__ == '__main__':
    # from unet import UNet
    from torchinfo import summary

    config = {'in_channels': 1,
              'out_channels': 1,
              'num_init_filters': 64,
              "structure": {
                  "dropout_rate": 0,
                  "conv_block": {
                      "normalization": "group_norm",
                      "activation_fct": "leakyReLU",
                      "kernal_size": 3,
                      "stride": 1,
                      "padding": 1
                  },
                  "pooling": {
                      "kernal_size": 2,
                      "stride": 2,
                      "padding": 0
                  },
                  "upconv": {
                      "kernal_size": 2,
                      "stride": 2,
                      "padding": 0
                  }
              }
              }

    model = ModUNet2D(config)

    # x = torch.randn(size=(20, 1, 512, 512), dtype=torch.float32)
    # with torch.no_grad():
    #     out, [enc_1, enc_2, enc_3, center] = model(x)
    #
    # print(f'Out: {out.shape}')
    # print(f'enc_1: {enc_1.shape}')
    # print(f'enc_2: {enc_2.shape}')
    # print(f'enc_3: {enc_3.shape}')
    # print(f'center: {center.shape}')

    batch_size = 2
    summary = summary(model, (batch_size, 1, 256, 256))
