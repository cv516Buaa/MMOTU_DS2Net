# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, build_upsample_layer

from ..builder import HEADS
from .decode_head import BaseDecodeHead
from ..utils import UpConvBlock
from ..backbones.unet import BasicConvBlock


@HEADS.register_module()
class TransUnetHead(BaseDecodeHead):
    """
    Args:
        trans_in_channels (int|Sequence[int]): Number of convs in the head. Default: [0]
        trans_in_index (int|Sequence[int]): Input vit feature index. Default: [0]
        trans_kernel_size (int): The kernel size for convs in the head. Default: 3.
        trans_num_convs (int): Number of convolutional layers in the
            convolution block of the correspondence decoder stage.
            Default: (2, 2, 2).
        trans_upsample_flag (Sequence[int]): Decide whether to use 
            Default: (True, True, True, True).
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
        conv_cfg (dict | None): Config dict for convolution layer.
            Default: None.
        norm_cfg (dict | None): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict | None): Config dict for activation layer in ConvModule.
            Default: dict(type='ReLU').
        upsample_cfg (dict): The upsample config of the upsample module in
            decoder. Default: dict(type='InterpConv').
        upsample_rate (int): The upsample times from the end feature to original images. Default: 2.
    """

    def __init__(self,
                 trans_kernel_size=3,
                 trans_num_convs=(2, 2, 2),
                 trans_upsample_flag=(False, True, True),
                 trans_dec_dilations=(1, 1, 1),
                 trans_in_channels=[768],
                 trans_in_index=[0],
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 upsample_cfg=dict(type='InterpConv'),
                 upsample_rate=2,
                 **kwargs):
        super(TransUnetHead, self).__init__(input_transform='multiple_select', **kwargs)
        ## first initialize BaseDecoderHead and generate res_in_channels and res_in_index 
        self.res_in_channels = self.in_channels
        self.res_in_index = self.in_index
        ## for multiple transformer output, first concat them together
        if len(trans_in_channels) > 1:
            self._init_inputs(trans_in_channels, trans_in_index, 'resize_concat')
        else:
            self._init_inputs(trans_in_channels, trans_in_index, self.input_transform)
        self.trans_in_channels = self.in_channels
        self.trans_in_index = self.in_index
        ## change in_index back
        self.in_index = self.res_in_index

        self.trans_num_convs = trans_num_convs
        self.trans_dec_dilations = trans_dec_dilations
        self.trans_upsample_flag = trans_upsample_flag

        assert len(self.trans_num_convs) == len(self.res_in_channels), \
            'The length of trans_num_convs should be equal to length of res_in_channels, '\
            f'while the trans_num_convs is {self.trans_num_convs}, the length of '\
            f'trans_num_convs is {len(self.trans_num_convs)}, and the sum channels is '\
            f'{len(self.res_in_channels)}.'
        
        ## init conv blocks for each stage
        self.num_stages = len(self.trans_num_convs)
        self.unet_decoder = nn.ModuleList()

        for i in range(self.num_stages):
            if i == 0:
                self.unet_decoder.append(
                    UpConvBlock(
                        conv_block=BasicConvBlock,
                        in_channels=self.trans_in_channels[0] if len(self.trans_in_channels)==1 else self.trans_in_channels,
                        skip_channels=self.res_in_channels[len(self.res_in_channels)-1],
                        out_channels=self.res_in_channels[len(self.res_in_channels)-1],
                        num_convs=self.trans_num_convs[0],
                        stride=1,
                        dilation=self.trans_dec_dilations[0],
                        with_cp=with_cp,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg,
                        upsample_cfg=upsample_cfg if self.trans_upsample_flag[0] else None,
                        dcn=None,
                        plugins=None))
            else:
                self.unet_decoder.append(
                    UpConvBlock(
                        conv_block=BasicConvBlock,
                        in_channels=self.res_in_channels[len(self.res_in_channels)-i],
                        skip_channels=self.res_in_channels[len(self.res_in_channels)-1-i],
                        out_channels=self.res_in_channels[len(self.res_in_channels)-1-i],
                        num_convs=self.trans_num_convs[i],
                        stride=1,
                        dilation=self.trans_dec_dilations[i],
                        with_cp=with_cp,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg,
                        upsample_cfg=upsample_cfg if self.trans_upsample_flag[i] else None,
                        dcn=None,
                        plugins=None))
        
        self.upsample_rate = upsample_rate
        self.upsample_end = nn.ModuleList()
        for i in range(self.upsample_rate):
            self.upsample_end.append(build_upsample_layer(
                cfg=upsample_cfg,
                in_channels=self.res_in_channels[0],
                out_channels=self.res_in_channels[0],
                with_cp=with_cp,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg))

    def forward(self, inputs):
        ## modified by LYU: 2022/04/12
        res_inputs, vit_inputs = inputs
        """Forward function."""
        res_x = self._transform_inputs(res_inputs)
        if len(self.trans_in_channels) > 1: 
            self.input_transform = 'resize_concat'
            vit_x = self._transform_inputs(vit_inputs)
        else:
            vit_x = vit_inputs[0]

        for i in range(self.num_stages):
            if i == 0:
                x = self.unet_decoder[0](res_x[len(self.res_in_channels)-1], vit_x)
            else:
                x = self.unet_decoder[i](res_x[len(self.res_in_channels)-1-i], x)

        for i in range(self.upsample_rate):
            x = self.upsample_end[i](x)
        output = self.cls_seg(x)
        return output
    
    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        """Forward function for training"""
        seg_logits = self.forward(inputs)
        losses = self.losses(seg_logits, gt_semantic_seg)
        return losses
    
    def forward_test(self, inputs, img_metas, test_cfg):
        """Forward function for testing."""
        return self.forward(inputs)
