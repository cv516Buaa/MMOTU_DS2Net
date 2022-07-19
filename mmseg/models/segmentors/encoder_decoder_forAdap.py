# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.core import add_prefix
from mmseg.ops import resize
from .. import builder
from ..builder import SEGMENTORS
from .base import BaseSegmentor


@SEGMENTORS.register_module()
class EncoderDecoder_forAdap(BaseSegmentor):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def __init__(self,
                 backbone,
                 decode_head,
                 discriminator=None,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(EncoderDecoder_forAdap, self).__init__(init_cfg)
        if pretrained is not None:
            assert backbone.get('pretrained') is None, \
                'both backbone and segmentor set pretrained weight'
            backbone.pretrained = pretrained
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        ## added by LYU: 2022/03/28
        self.discriminators = builder.build_discriminator(discriminator)
        self._parse_train_cfg()

        assert self.with_decode_head
    
    def _parse_train_cfg(self):
        """Parsing train config and set some attributes for training."""
        if self.train_cfg is None:
            self.train_cfg = dict()
        # control the work flow in train step
        self.disc_steps = self.train_cfg.get('disc_steps', 1)

        self.disc_init_steps = (0 if self.train_cfg is None else
                                self.train_cfg.get('disc_init_steps', 0))

        self.real_img_key = self.train_cfg.get('real_img_key', 'real_img')

    def _init_decode_head(self, decode_head):
        """Initialize ``decode_head``"""
        self.decode_head = builder.build_head(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes

    def _init_auxiliary_head(self, auxiliary_head):
        """Initialize ``auxiliary_head``"""
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(builder.build_head(head_cfg))
            else:
                self.auxiliary_head = builder.build_head(auxiliary_head)

    def extract_feat(self, img):
        """Extract features from images."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def encode_decode(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x = self.extract_feat(img)
        out = self._decode_head_forward_test(x, img_metas)
        out = resize(
            input=out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        return out

    def _decode_head_forward_train(self, x, img_metas, gt_semantic_seg):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode = self.decode_head.forward_train(x, img_metas,
                                                     gt_semantic_seg,
                                                     self.train_cfg)

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def _decode_head_forward_test(self, x, img_metas):
        """Run forward function and calculate loss for decode head in
        inference."""
        seg_logits = self.decode_head.forward_test(x, img_metas, self.test_cfg)
        return seg_logits

    def _auxiliary_head_forward_train(self, x, img_metas, gt_semantic_seg):
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux = aux_head.forward_train(x, img_metas,
                                                  gt_semantic_seg,
                                                  self.train_cfg)
                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
        else:
            loss_aux = self.auxiliary_head.forward_train(
                x, img_metas, gt_semantic_seg, self.train_cfg)
            losses.update(add_prefix(loss_aux, 'aux'))

        return losses

    def forward_dummy(self, img):
        """Dummy forward function."""
        seg_logit = self.encode_decode(img, None)

        return seg_logit

    def forward_train(self, img, img_metas, gt_semantic_seg, B_img):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        x = self.extract_feat(img)
        
        losses = dict()

        loss_decode = self._decode_head_forward_train(x, img_metas,
                                                      gt_semantic_seg)
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                x, img_metas, gt_semantic_seg)
            losses.update(loss_aux)

        return losses
    
    def segmentor_forward(self, img, B_img):
        outputs = dict()
        fb_s = self.extract_feat(img)
        fb_t = self.extract_feat(B_img)
        pred_s = self.decode_head(fb_s)
        pred_t = self.decode_head(fb_t)
        outputs['pred_s'] = pred_s
        outputs['pred_t'] = pred_t
        pred_s_aux = []
        if self.with_auxiliary_head:
            if isinstance(self.auxiliary_head, nn.ModuleList):
                for idx, aux_head in enumerate(self.auxiliary_head):
                    pred_s_aux.append(aux_head(fb_s))
            else:
                pred_s_aux = self.auxiliary_head(fb_s)
            outputs['pred_s_aux'] = pred_s_aux
        return outputs
    
    def discriminator_forward(self, seg_pred):
        outputs = dict()
        outputs['pred_s_dis'] = self.discriminators(seg_pred['pred_s'])
        outputs['pred_t_dis'] = self.discriminators(seg_pred['pred_t'])
        ## added by LYU: 2022/04/01 only support one discriminator
        ## auxiliary_discriminator: TBD
        return outputs

    def _get_segmentor_loss(self, outputs, gt_semantic_seg):
        losses = dict()
        loss_seg = self.decode_head.losses(outputs['pred_s'], gt_semantic_seg)
        losses.update(loss_seg)
        if self.with_auxiliary_head:
            if isinstance(self.auxiliary_head, nn.ModuleList):
                for idx, aux_head in enumerate(self.auxiliary_head):
                    loss_aux = aux_head.losses(outputs['pred_s_aux'][idx], gt_semantic_seg)
                    losses.update(add_prefix(loss_aux, f'aux_{idx}'))
            else:
                loss_aux = self.auxiliary_head.losses(outputs['pred_s_aux'], gt_semantic_seg)
                losses.update(add_prefix(loss_aux, 'aux'))
        loss_seg, log_vars_seg = self._parse_losses(losses)
        return loss_seg, log_vars_seg
    
    ## added by LYU: 2022/04/06
    def _get_gan_loss(self, pred, domain, target_is_real):
        losses = dict()
        losses[f'loss_gan_{domain}'] = self.discriminators.gan_loss(pred, target_is_real)
        loss_g, log_vars_g = self._parse_losses(losses)
        ## added by LYU: 2022/04/06 only support one gan_loss
        ## auxiliary_ganloss: TBD
        return loss_g, log_vars_g

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

        optimizer['backbone'].zero_grad()
        optimizer['decode_head'].zero_grad()
        ## added by LYU: 2022/04/11
        if self.with_auxiliary_head:
            optimizer['auxiliary_head'].zero_grad()
        optimizer['discriminators'].zero_grad()
        log_vars = dict()

        # train segmentor with source and target
        # discriminator weights shut down
        # 1. train segmentor with source
        self.set_requires_grad(self.discriminators, False)
        seg_outputs = self.segmentor_forward(data_batch['img'], data_batch['B_img'])
        loss_seg, log_vars_seg = self._get_segmentor_loss(seg_outputs, data_batch['gt_semantic_seg'])
        loss_seg.backward()
        log_vars.update(log_vars_seg)
        
        # 2. train segmentor with target
        # add by LYU: 2022/04/11: for multi-outputs decoder head
        seg_output_adv = dict()
        if isinstance(seg_outputs['pred_t'], tuple):
            #seg_pred_t = F.softmax(seg_outputs['pred_t'][0])
            #seg_pred_s = F.softmax(seg_outputs['pred_s'][0])
            ## modified by LYU: 2022/04/18
            seg_pred_t = self.sw_softmax(seg_outputs['pred_t'][0])
            seg_pred_s = self.sw_softmax(seg_outputs['pred_s'][0])
        else:
            #seg_pred_t = F.softmax(seg_outputs['pred_t'])
            #seg_pred_s = F.softmax(seg_outputs['pred_s'])
            ## modified by LYU: 2022/04/18
            seg_pred_t = self.sw_softmax(seg_outputs['pred_t'])
            seg_pred_s = self.sw_softmax(seg_outputs['pred_s'])
        seg_output_adv['pred_t'] = seg_pred_t
        seg_output_adv['pred_s'] = seg_pred_s
        dis_outputs = self.discriminator_forward(seg_output_adv)
        dis_outputs['pred_t_dis'] = resize(
            input=dis_outputs['pred_t_dis'],
            size=data_batch['B_img'].shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        dis_outputs['pred_s_dis'] = resize(
            input=dis_outputs['pred_s_dis'],
            size=data_batch['img'].shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        loss_adv, log_vars_adv = self._get_gan_loss(dis_outputs['pred_t_dis'], 'pred_t_dis_seg', 1)
        loss_adv.backward()
        log_vars.update(log_vars_adv)

        optimizer['backbone'].step()
        optimizer['decode_head'].step()
        ## added by LYU: 2022/04/11
        if self.with_auxiliary_head:
            optimizer['auxiliary_head'].step()

        # 3. train discriminator with source
        self.set_requires_grad(self.discriminators, True)
        pred_t_dis = seg_outputs['pred_t'].detach()
        pred_s_dis = seg_outputs['pred_s'].detach()
        #pred_t_dis = F.softmax(pred_t_dis)
        #pred_s_dis = F.softmax(pred_s_dis)
        ## modified by LYU: 2022/04/18
        pred_t_dis = self.sw_softmax(pred_t_dis)
        pred_s_dis = self.sw_softmax(pred_s_dis)
        seg_outputs_detach = dict()
        seg_outputs_detach['pred_t'] = pred_t_dis
        seg_outputs_detach['pred_s'] = pred_s_dis
        dis_outputs = self.discriminator_forward(seg_outputs_detach)
        dis_outputs['pred_t_dis'] = resize(
            input=dis_outputs['pred_t_dis'],
            size=data_batch['B_img'].shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        dis_outputs['pred_s_dis'] = resize(
            input=dis_outputs['pred_s_dis'],
            size=data_batch['img'].shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        loss_adv_ds, log_vars_adv_ds = self._get_gan_loss(dis_outputs['pred_s_dis'], 'pred_s_dis_d', 1)
        loss_adv_ds.backward()
        log_vars.update(log_vars_adv_ds)

        # 4. train discriminator with target
        loss_adv_dt, log_vars_adv_dt = self._get_gan_loss(dis_outputs['pred_t_dis'], 'pred_t_dis_d', 0)
        loss_adv_dt.backward()
        log_vars.update(log_vars_adv_dt)
        
        optimizer['discriminators'].step()
        # train discriminator 
        loss = loss_seg
        if hasattr(self, 'iteration'):
            self.iteration += 1
        '''
        losses = self(**data_batch)
        loss, log_vars = self._parse_losses(losses)
        '''
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
    
    @staticmethod
    def sw_softmax(pred):
        N, C, H, W = pred.shape
        pred_sh = torch.reshape(pred, (N, C, H*W))
        pred_sh = F.softmax(pred_sh, dim=2)
        pred_out = torch.reshape(pred_sh, (N, C, H, W))
        return pred_out