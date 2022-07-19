from .resnet import *
from .vit import TransformerEncoderLayer
from ..utils import PatchEmbed
from mmseg.ops import resize

from torch.nn.modules.utils import _pair as to_2tuple
import torch
from mmcv.runner import ModuleList
from mmcv.cnn import build_norm_layer

@BACKBONES.register_module()
class ResNetV1c_vit(ResNet):

    def __init__(self, 
                 vit_input_indices=[0, 1, 2, 3],
                 vit_output_indices=-1,
                 vit_in_channels=1024,
                 vit_embed_dims=768,
                 vit_patch_size=1, 
                 vit_norm_cfg=dict(type='LN'),
                 vit_patch_norm=False, 
                 vit_feature_size=32, 
                 vit_with_cls_token=True,
                 vit_output_cls_token=False,
                 vit_drop_rate=0.0,
                 vit_num_layers=12, 
                 vit_drop_path_rate=0.0,
                 vit_num_heads=12,
                 vit_mlp_ratio=4,
                 vit_attn_drop_rate=0.0,
                 vit_num_fcs=2,
                 vit_qkv_bias=True,
                 vit_act_cfg=dict(type='GELU'),
                 vit_final_norm=False,
                 vit_interpolate_mode='bicubic',
                 **kwargs):
        super(ResNetV1c_vit, self).__init__(
            deep_stem=True, avg_down=False, **kwargs)
        """ResNetV1c_vit backbone.
        Args:
        vit_input_indices (Sequence[int]): Decide which stage of ResNet for vit input, Default: (0, 1, 2, 3).
        vit_output_indices (Sequence[int]): Decide which stage of vit for output, Default: -1
        vit_in_channels (int): Number of input image/feature channels. Default: 1024.
        vit_embed_dims (int): dimension of embed feature after patching. Default: 768.
        vit_patch_size (int): size of patch on feature map. Default: 1.
        vit_norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN')
        vit_patch_norm (bool): Whether to add a norm in PatchEmbed Block.
            Default: False.
        vit_feature_size (int | tuple): feature map size. Default: 32.
        vit_drop_rate (float): Probability of an element to be zeroed
            after the feed forward layer. Default: 0.0.
        vit_attn_drop_rate (float): The drop out rate for attention layer.
            Default: 0.0.
        vit_num_fcs (int): The number of fully-connected layers for FFNs.
            Default: 2.
        vit_qkv_bias (bool): enable bias for qkv if True. Default: True
        vit_act_cfg (dict): The activation config for FFNs.
            Default: dict(type='GELU').
        vit_mlp_ratio (int): ratio of mlp hidden dim to embedding dim.
            Default: 4.
        vit_interpolate_mode (str): Select the interpolate mode for position
            embeding vector resize. Default: bicubic.
        """
        ## vit part
        if isinstance(vit_feature_size, int):
            vit_feature_size = to_2tuple(vit_feature_size)
        elif isinstance(vit_feature_size, tuple):
            if len(vit_feature_size) == 1:
                vit_feature_size = to_2tuple(vit_feature_size[0])
            assert len(vit_feature_size) == 2, \
                f'The size of feature map should have length 1 or 2, ' \
                f'but got {len(vit_feature_size)}'
        
        if isinstance(vit_output_indices, int):
            if vit_output_indices == -1:
                vit_output_indices = vit_num_layers - 1
            self.vit_output_indices = [vit_output_indices]
        elif isinstance(vit_output_indices, list) or isinstance(vit_output_indices, tuple):
            self.vit_output_indices = vit_output_indices
        else:
            raise TypeError('vit_output_indices must be type of int, list or tuple')
        
        self.vit_feature_size = vit_feature_size
        self.vit_input_indices = vit_input_indices
        self.vit_in_channels = vit_in_channels
        self.vit_embed_dims = vit_embed_dims
        self.vit_patch_size = vit_patch_size
        self.vit_norm_cfg = vit_norm_cfg
        self.vit_patch_norm = vit_patch_norm
        self.vit_with_cls_token = vit_with_cls_token
        self.vit_output_cls_token = vit_output_cls_token
        self.vit_drop_rate = vit_drop_rate
        self.vit_drop_path_rate = vit_drop_path_rate
        self.vit_num_layers = vit_num_layers
        self.vit_num_heads = vit_num_heads
        self.vit_mlp_ratio = vit_mlp_ratio
        self.vit_attn_drop_rate = vit_attn_drop_rate
        self.vit_num_fcs = vit_num_fcs
        self.vit_qkv_bias = vit_qkv_bias
        self.vit_act_cfg = vit_act_cfg
        self.vit_final_norm = vit_final_norm
        self.vit_interpolate_mode = vit_interpolate_mode
        self.patch_embed = PatchEmbed(
            in_channels=self.vit_in_channels,
            embed_dims=self.vit_embed_dims,
            conv_type='Conv2d',
            kernel_size=self.vit_patch_size,
            stride=self.vit_patch_size,
            padding='corner',
            norm_cfg=self.vit_norm_cfg if self.vit_patch_norm else None,
            init_cfg=None,
        )
        num_patches = (vit_feature_size[0] // vit_patch_size) * \
            (vit_feature_size[1] // vit_patch_size)

        self.vit_cls_token = nn.Parameter(torch.zeros(1, 1, self.vit_embed_dims))
        self.vit_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, self.vit_embed_dims))
        self.vit_drop_after_pos = nn.Dropout(p=self.vit_drop_rate)

        dpr = [
            x.item() for x in torch.linspace(0, self.vit_drop_path_rate, self.vit_num_layers)
        ]  # stochastic depth decay rule
        self.vit_layers = ModuleList()

        for i in range(self.vit_num_layers):
            self.vit_layers.append(
                TransformerEncoderLayer(
                    embed_dims=self.vit_embed_dims,
                    num_heads=self.vit_num_heads,
                    feedforward_channels=self.vit_mlp_ratio * self.vit_embed_dims,
                    attn_drop_rate=self.vit_attn_drop_rate,
                    drop_rate=self.vit_drop_rate,
                    drop_path_rate=dpr[i],
                    num_fcs=self.vit_num_fcs,
                    qkv_bias=self.vit_qkv_bias,
                    act_cfg=self.vit_act_cfg,
                    norm_cfg=self.vit_norm_cfg,
                    batch_first=True))
        
        if self.vit_final_norm:
            self.norm1_name, norm1 = build_norm_layer(
                norm_cfg, embed_dims, postfix=1)
            self.add_module(self.norm1_name, norm1)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    def _pos_embeding(self, patched_img, hw_shape, pos_embed):
        """Positiong embeding method.

        Resize the pos_embed, if the input image size doesn't match
            the training size.
        Args:
            patched_img (torch.Tensor): The patched image, it should be
                shape of [B, L1, C].
            hw_shape (tuple): The downsampled image resolution.
            pos_embed (torch.Tensor): The pos_embed weighs, it should be
                shape of [B, L2, c].
        Return:
            torch.Tensor: The pos encoded image feature.
        """
        assert patched_img.ndim == 3 and pos_embed.ndim == 3, \
            'the shapes of patched_img and pos_embed must be [B, L, C]'
        x_len, pos_len = patched_img.shape[1], pos_embed.shape[1]
        if x_len != pos_len:
            if pos_len == (self.vit_feature_size[0] // self.vit_patch_size) * (
                    self.vit_feature_size[1] // self.vit_patch_size) + 1:
                pos_h = self.vit_feature_size[0] // self.vit_patch_size
                pos_w = self.vit_feature_size[1] // self.vit_patch_size
            else:
                raise ValueError(
                    'Unexpected shape of pos_embed, got {}.'.format(
                        pos_embed.shape))
            pos_embed = self.resize_pos_embed(pos_embed, hw_shape,
                                              (pos_h, pos_w),
                                              self.vit_interpolate_mode)
        return self.vit_drop_after_pos(patched_img + pos_embed)

    @staticmethod
    def resize_pos_embed(pos_embed, input_shape, pos_shape, mode):
        """Resize pos_embed weights.

        Resize pos_embed using bicubic interpolate method.
        Args:
            pos_embed (torch.Tensor): Position embedding weights.
            input_shape (tuple): Tuple for (downsampled input image height,
                downsampled input image width).
            pos_shape (tuple): The resolution of downsampled origin training
                image.
            mode (str): Algorithm used for upsampling:
                ``'nearest'`` | ``'linear'`` | ``'bilinear'`` | ``'bicubic'`` |
                ``'trilinear'``. Default: ``'nearest'``
        Return:
            torch.Tensor: The resized pos_embed of shape [B, L_new, C]
        """
        assert pos_embed.ndim == 3, 'shape of pos_embed must be [B, L, C]'
        pos_h, pos_w = pos_shape
        cls_token_weight = pos_embed[:, 0]
        pos_embed_weight = pos_embed[:, (-1 * pos_h * pos_w):]
        pos_embed_weight = pos_embed_weight.reshape(
            1, pos_h, pos_w, pos_embed.shape[2]).permute(0, 3, 1, 2)
        pos_embed_weight = resize(
            pos_embed_weight, size=input_shape, align_corners=False, mode=mode)
        cls_token_weight = cls_token_weight.unsqueeze(1)
        pos_embed_weight = torch.flatten(pos_embed_weight, 2).transpose(1, 2)
        pos_embed = torch.cat((cls_token_weight, pos_embed_weight), dim=1)
        return pos_embed
    
    def forward(self, x):
        """Forward function."""
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)
        x = self.maxpool(x)
        outs_res = []
        outs_vit = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs_res.append(x)
            if i in self.vit_input_indices:
                B = x.shape[0]
                x_vit, hw_shape = self.patch_embed(x)
                cls_tokens = self.vit_cls_token.expand(B, -1, -1)
                x_vit = torch.cat((cls_tokens, x_vit), dim=1)
                x_vit = self._pos_embeding(x_vit, hw_shape, self.vit_pos_embed)
                if not self.vit_with_cls_token:
                    # Remove class token for transformer encoder input
                    x_vit = x_vit[:, 1:]
                for j, vit_layer in enumerate(self.vit_layers):
                    x_vit = vit_layer(x_vit)
                    if j == len(self.vit_layers) - 1:
                        if self.vit_final_norm:
                            x_vit = self.norm1(x_vit)
                    if j in self.vit_output_indices:
                        if self.vit_with_cls_token:
                            # Remove class token and reshape token for decoder head
                            out_vit = x_vit[:, 1:]
                        else:
                            out_vit = x_vit
                        B, _, C = out_vit.shape
                        out_vit = out_vit.reshape(B, hw_shape[0], hw_shape[1],
                                  C).permute(0, 3, 1, 2).contiguous()
                        if self.vit_output_cls_token:
                            out_vit = [out_vit, x_vit[:, 0]]
                        outs_vit.append(out_vit)
        return tuple(outs_res), tuple(outs_vit)