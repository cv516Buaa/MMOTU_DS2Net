# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.core import add_prefix
from mmseg.ops import resize
from .. import builder
from ..builder import SEGMENTORS
from .base import BaseSegmentor

import numpy as np
import copy

import matplotlib.pyplot as plt

@SEGMENTORS.register_module()
class EncoderDecoder_forDAFormer(BaseSegmentor):
    """Encoder Decoder segmentors for DAFormer.

    EncoderDecoder_forDAFormer typically consists of two backbone, two decode_head. Here, we do not
    apply auxiliary_head, neck to simplify the implementation.

    """

    def __init__(self,
                 backbone,
                 decode_head,
                 cross_EMA= None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(EncoderDecoder_forDAFormer, self).__init__(init_cfg)
        if pretrained is not None:
            assert backbone.get('pretrained') is None, \
                'both backbone and segmentor set pretrained weight'
            backbone.pretrained = pretrained
        self.backbone = builder.build_backbone(backbone)
        self.decode_head = self._init_decode_head(decode_head)
        self.num_classes = self.decode_head.num_classes
        self.align_corners = self.decode_head.align_corners

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        ## added by LYU: 2022/11/22
        if cross_EMA is not None:
            self.cross_EMA = cross_EMA
            self._init_cross_EMA(self.cross_EMA)
        self._parse_train_cfg()
    
    ##############################
    ## added by LYU: 2022/11/22
    ## added for cross_EMA
    def _init_cross_EMA(self, cfg):
        self.cross_EMA_type = cfg['type']
        self.cross_EMA_alpha = cfg['decay']
        self.cross_EMA_training_ratio = cfg['training_ratio']
        self.cross_EMA_pseu_cls_weight = cfg['pseudo_class_weight']
        self.cross_EMA_pseu_thre = cfg['pseudo_threshold']
        self.cross_EMA_rare_pseu_thre = cfg['pseudo_rare_threshold']
        if self.cross_EMA_type == 'single_t':
            self.cross_EMA_backbone = builder.build_backbone(cfg['backbone_EMA'])
            self.cross_EMA_decoder = self._init_decode_head(cfg['decode_head_EMA'])
        elif self.cross_EMA_type == 'single_decoder':
            self.cross_EMA_decoder = self._init_decode_head(cfg['decode_head_EMA'])
        else:
            ## No cross_EMA
            pass
        
    def _update_cross_EMA(self, iter):
        alpha_t = min(1 - 1 / (iter + 1), self.cross_EMA_alpha)
        if self.cross_EMA_type == 'single_decoder':
            pass
        if self.cross_EMA_type == 'single_t':
            ## 1. update target_backbone
            for ema_b, target_b in zip(self.cross_EMA_backbone.parameters(), self.backbone.parameters()):
                ## For scalar params
                if not target_b.data.shape:
                    ema_b.data = alpha_t * ema_b.data + (1 - alpha_t) * target_b.data
                ## For tensor params
                else:
                    ema_b.data[:] = alpha_t * ema_b.data[:] + (1 - alpha_t) * target_b.data[:]

            ## 2. updata target_decoder
            for ema_d, target_d in zip(self.cross_EMA_decoder.parameters(), self.decode_head.parameters()):
                ## For scalar params
                if not target_d.data.shape:
                    ema_d.data = alpha_t * ema_d.data + (1 - alpha_t) * target_d.data
                ## For tensor params
                else:
                    ema_d.data[:] = alpha_t * ema_d.data[:] + (1 - alpha_t) * target_d.data[:]
    
    def pseudo_label_generation_crossEMA(self, pred, dev=None):
        ##############################
        #### 1. vanilla pseudo label generation
        pred_softmax = torch.softmax(pred, dim=1)
        pseudo_prob, pseudo_label = torch.max(pred_softmax, dim=1)
        ps_large_p = pseudo_prob.ge(self.cross_EMA_pseu_thre).long() == 1
        ps_size = np.size(np.array(pseudo_label.cpu()))
        pseudo_weight_ratio = torch.sum(ps_large_p).item() / ps_size
        pseudo_weight = pseudo_weight_ratio * torch.ones(pseudo_prob.shape, device=dev)
        ##############################
        ##############################
        #### 2. class balancing strategy
        #### 2.1 change pseudo_weight and further set a threshold for rare class. E.g. For threshold over 0.8: 10x for car and clutter; 5x for 'low_vegetation' and 'tree'
        if self.cross_EMA_pseu_cls_weight is not None and self.cross_EMA_rare_pseu_thre is not None:
            ps_large_p_rare = pseudo_prob.ge(self.cross_EMA_rare_pseu_thre).long() == 1
            pseudo_weight = pseudo_weight * ps_large_p_rare
            pseudo_class_weight = copy.deepcopy(pseudo_label.float())
            for i in range(len(self.cross_EMA_pseu_cls_weight)):
                pseudo_class_weight[pseudo_class_weight == i] = self.cross_EMA_pseu_cls_weight[i]
            pseudo_weight = pseudo_class_weight * pseudo_weight
            pseudo_weight[pseudo_weight == 0] = pseudo_weight_ratio * 0.5
        ##############################
        pseudo_label = pseudo_label[:, None, :, :]
        return pseudo_label, pseudo_weight

    def encode_decode_crossEMA(self, input=None, dev=None):
        ## option1: 'single_t': inference all EMA_teacher including EMA_backbone and cross_EMA_decoder
        if self.cross_EMA_type == 'single_decoder':
            pass
        if self.cross_EMA_type == 'single_t':
            """Encode images with backbone and decode into a semantic segmentation map of the same size as input."""
            ## 1. forward backbone
            F_t = self.forward_backbone(self.backbone, input)
            ## 2. forward decode_head
            P_t = self.forward_decode_head(self.decode_head, F_t)

            P_EMA = resize(
                input=P_t,
                size=input.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            
            ## 3. pseudo label generation
            P_EMA_detach = P_EMA.detach()
            pseudo_label, pseudo_weight = self.pseudo_label_generation_crossEMA(P_EMA_detach, dev)
        return pseudo_label, pseudo_weight
    
    ## CODE for EMA
    ##############################

    def _parse_train_cfg(self):
        """Parsing train config and set some attributes for training."""
        if self.train_cfg is None:
            self.train_cfg = dict()
        # control the work flow in train step
        self.disc_steps = self.train_cfg.get('disc_steps', 1)

        self.disc_init_steps = (0 if self.train_cfg is None else
                                self.train_cfg.get('disc_init_steps', 0))

    ## modified by LYU: 2022/04/22
    def _init_decode_head(self, decode_head):
        """Initialize ``decode_head``"""
        decode_head = builder.build_head(decode_head)
        return decode_head

    def extract_feat(self, img):
        """Extract features from images."""
        x = self.backbone_s(img)
        return x

    def encode_decode(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        ## 1. forward backbone
        F = self.forward_backbone(self.backbone, img)
      
        ## 3. forward decode_head
        P = self.forward_decode_head(self.decode_head, F)
        out = resize(
            input=P,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        return out

    def _decode_head_forward_test(self, x, img_metas):
        """Run forward function and calculate loss for decode head in
        inference."""
        seg_logits = self.decode_head.forward_test(x, img_metas, self.test_cfg)
        return seg_logits

    def forward_dummy(self, img):
        """Dummy forward function."""
        seg_logit = self.encode_decode(img, None)

        return seg_logit

    ## added by LYU: 2022/05/05
    def forward_backbone(self, backbone, img):
        F_b = backbone(img)
        return F_b
    
    def forward_decode_head(self, decode_head, feature):
        Pred = decode_head(feature)
        return Pred

    def forward_train(self, img, B_img):
        pass
        """Forward function for training."""

    def _get_segmentor_loss(self, decode_head, pred, gt_semantic_seg, gt_weight=None):
        losses = dict()
        loss_seg = decode_head.losses(pred, gt_semantic_seg, gt_weight=gt_weight)
        losses.update(loss_seg)
        loss_seg, log_vars_seg = self._parse_losses(losses)
        return loss_seg, log_vars_seg

    def train_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during training.

        The whole process including back propagation and 
        optimizer updating is also defined in this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """
        ## added by LYU: 2022/03/29
        # dirty walkround for not providing running status
        if not hasattr(self, 'iteration'):
            self.iteration = 0
        curr_iter = self.iteration
        
        ## added by LYU: 2022/11/23
        ## CODE for EMA
        if curr_iter > 0:
            self._update_cross_EMA(curr_iter)

        ## 1. towards all optimizers, clear gradients
        optimizer['backbone'].zero_grad()
        optimizer['decode_head'].zero_grad()

        log_vars = dict()

        ## 1.1 forward backbone
        F_s = self.forward_backbone(self.backbone, data_batch['img'])
        F_t = self.forward_backbone(self.backbone, data_batch['B_img'])

        ## 1.2 forward head
        P_s = self.forward_decode_head(self.decode_head, F_s)
        P_t = self.forward_decode_head(self.decode_head, F_t)
        loss_seg_s, log_vars_seg_s = self._get_segmentor_loss(self.decode_head, P_s, data_batch['gt_semantic_seg'])
        log_vars.update(log_vars_seg_s)
        loss_seg = loss_seg_s

        ##############################
        ## 1.2 forward EMA for pseudo_label
        ## CODE for EMA
        # FOR decode_only_t
        pseudo_label, pseudo_weight = self.encode_decode_crossEMA(input=data_batch['B_img'], dev=data_batch['img'].device)
        loss_seg_t, log_vars_seg_t = self._get_segmentor_loss(self.decode_head, P_t, pseudo_label, gt_weight=pseudo_weight)
        log_vars_seg_t['loss_ce_seg_t'] = log_vars_seg_t.pop('loss_ce')
        log_vars_seg_t['acc_seg_t'] = log_vars_seg_t.pop('acc_seg')
        log_vars_seg_t['loss_ce_seg_t'] = log_vars_seg_t.pop('loss')
        log_vars.update(log_vars_seg_t)
   
        loss_seg = loss_seg + self.cross_EMA_training_ratio * loss_seg_t
        ## CODE for EMA
        ##############################
        loss_seg.backward()
        optimizer['backbone'].step()
        optimizer['decode_head'].step()
       
        loss = loss_seg
        if hasattr(self, 'iteration'):
            self.iteration += 1        
        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(data_batch['img_metas']))

        return outputs

    # TODO refactor
    def slide_inference(self, img, img_meta, rescale):
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """

        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = img.size()
        num_classes = self.num_classes
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, num_classes, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                crop_seg_logit = self.encode_decode(crop_img, img_meta)
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        if torch.onnx.is_in_onnx_export():
            # cast count_mat to constant while exporting to ONNX
            count_mat = torch.from_numpy(
                count_mat.cpu().detach().numpy()).to(device=img.device)
        preds = preds / count_mat
        if rescale:
            preds = resize(
                preds,
                size=img_meta[0]['ori_shape'][:2],
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)
        return preds

    def whole_inference(self, img, img_meta, rescale):
        """Inference with full image."""

        seg_logit = self.encode_decode(img, img_meta)
        if rescale:
            # support dynamic shape for onnx
            if torch.onnx.is_in_onnx_export():
                size = img.shape[2:]
            else:
                size = img_meta[0]['ori_shape'][:2]
            seg_logit = resize(
                seg_logit,
                size=size,
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)

        return seg_logit

    def inference(self, img, img_meta, rescale):
        """Inference with slide/whole style.

        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            img_meta (dict): Image info dict where each dict has: 'img_shape',
                'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            rescale (bool): Whether rescale back to original shape.

        Returns:
            Tensor: The output segmentation map.
        """

        assert self.test_cfg.mode in ['slide', 'whole']
        ori_shape = img_meta[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in img_meta)
        if self.test_cfg.mode == 'slide':
            seg_logit = self.slide_inference(img, img_meta, rescale)
        else:
            seg_logit = self.whole_inference(img, img_meta, rescale)
        output = F.softmax(seg_logit, dim=1)
        flip = img_meta[0]['flip']
        if flip:
            flip_direction = img_meta[0]['flip_direction']
            assert flip_direction in ['horizontal', 'vertical']
            if flip_direction == 'horizontal':
                output = output.flip(dims=(3, ))
            elif flip_direction == 'vertical':
                output = output.flip(dims=(2, ))

        return output

    def simple_test(self, img, img_meta, rescale=True):
        """Simple test with single image."""
        seg_logit = self.inference(img, img_meta, rescale)
        seg_pred = seg_logit.argmax(dim=1)
        if torch.onnx.is_in_onnx_export():
            # our inference backend only support 4D output
            seg_pred = seg_pred.unsqueeze(0)
            return seg_pred
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred

    def aug_test(self, imgs, img_metas, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented seg logit inplace
        seg_logit = self.inference(imgs[0], img_metas[0], rescale)
        for i in range(1, len(imgs)):
            cur_seg_logit = self.inference(imgs[i], img_metas[i], rescale)
            seg_logit += cur_seg_logit
        seg_logit /= len(imgs)
        seg_pred = seg_logit.argmax(dim=1)
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred

    @staticmethod
    def set_requires_grad(nets, requires_grad=False):
        """Set requires_grad for all the networks.

        Args:
            nets (nn.Module | list[nn.Module]): A list of networks or a single
                network.
            requires_grad (bool): Whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad