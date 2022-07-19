import torch.nn as nn
from ..builder import DISCRIMINATORS, build_loss

from mmcv.cnn import ConvModule, build_conv_layer

@DISCRIMINATORS.register_module()
class AdapSegDiscriminator(nn.Module):
    def __init__(self,
                 in_channels,
                 base_channels=64,
                 num_conv=3,
                 gan_loss=None,
                 norm_cfg=dict(type='BN'),
                 init_cfg=dict(type='normal', gain=0.02)):
        super().__init__()

        # support no gan_loss in testing
        if gan_loss is not None:
            self.gan_loss = build_loss(gan_loss)
        else:
            self.gan_loss = None

        kernel_size = 4
        padding = 1

        # input layer
        sequence = [
            ConvModule(
                in_channels=in_channels,
                out_channels=base_channels,
                kernel_size=kernel_size,
                stride=2,
                padding=padding,
                bias=True,
                norm_cfg=None,
                act_cfg=dict(type='LeakyReLU', negative_slope=0.2))
        ]

        # stacked intermediate layers,
        # gradually increasing the number of filters
        multiple_now = 1
        multiple_prev = 1
        for n in range(1, num_conv):
            multiple_prev = multiple_now
            multiple_now = min(2**n, 8)
            sequence += [
                ConvModule(
                    in_channels=base_channels * multiple_prev,
                    out_channels=base_channels * multiple_now,
                    kernel_size=kernel_size,
                    stride=2,
                    padding=padding,
                    bias=True,
                    norm_cfg=norm_cfg,
                    act_cfg=dict(type='LeakyReLU', negative_slope=0.2))
            ]
        
        multiple_prev = multiple_now
        multiple_now = min(2**num_conv, 8)
        sequence += [
            ConvModule(
                in_channels=base_channels * multiple_prev,
                out_channels=base_channels * multiple_now,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                bias=True,
                norm_cfg=norm_cfg,
                act_cfg=dict(type='LeakyReLU', negative_slope=0.2))
        ]
        
        # output one-channel prediction map
        sequence += [
            build_conv_layer(
                dict(type='Conv2d'),
                base_channels * multiple_now,
                1,
                kernel_size=kernel_size,
                stride=1,
                padding=padding)
        ]
        self.discriminator = nn.Sequential(*sequence)
    
    def forward(self, x):
        return self.discriminator(x)