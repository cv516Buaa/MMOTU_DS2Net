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
class EncoderDecoder_forDS2Net(BaseSegmentor):
    """Encoder Decoder segmentors for DSFN.

    EncoderDecoder_forDSFN typically consists of two backbone, two decode_head. Here, we do not
    apply auxiliary_head, neck to simplify the implementation.

    Args:
        backbone_s: backbone for source.
        backbone_t: backbone for target.
        decode_head_s: decode_head for source
        decode_head_t: decode_head for target
        discriminator_s: discriminator for source and fake_source
        discriminator_t: discriminator for target and fake_target
        discriminator_fs: discriminator for source_prediction and fake_source_prediction
        discriminator_ft: discriminator for target_prediction and fake_target_prediction
    """

    def __init__(self,
                 backbone_s,
                 backbone_t,
                 decode_head_s,
                 decode_head_t,
                 discriminator_s=None,
                 discriminator_t=None,
                 discriminator_fs=None,
                 discriminator_ft=None,
                 dsk_neck=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(EncoderDecoder_forDS2Net, self).__init__(init_cfg)
        if pretrained is not None:
            assert backbone_s.get('pretrained') is None, \
                'both backbone_s and segmentor set pretrained weight'
            assert backbone_t.get('pretrained') is None, \
                'both backbone_t and segmentor set pretrained weight'
            backbone_s.pretrained = pretrained
            backbone_t.pretrained = pretrained
        self.backbone_s = builder.build_backbone(backbone_s)
        self.backbone_t = builder.build_backbone(backbone_t)
        
        self.decode_head_s = self._init_decode_head(decode_head_s)
        self.decode_head_t = self._init_decode_head(decode_head_t)
        self.num_classes = self.decode_head_t.num_classes
        self.align_corners = self.decode_head_t.align_corners
        assert self.decode_head_s.num_classes == self.decode_head_t.num_classes, \
                'both decode_head_s and decode_head_t must have same num_classes'

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        
        ## added by LYU: 2022/04/22
        self.discriminator_s = builder.build_discriminator(discriminator_s)
        self.discriminator_t = builder.build_discriminator(discriminator_t)
        self.discriminator_fs = builder.build_discriminator(discriminator_fs)
        self.discriminator_ft = builder.build_discriminator(discriminator_ft)
        ## added by LYU: 2022/05/05
        self.dsk_neck = builder.build_neck(dsk_neck)
        self._parse_train_cfg()
    
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
        if isinstance(self.dsk_neck.in_channels, int):
            F_t2s = self.forward_backbone(self.backbone_s, img)[-1]
            F_t2t = self.forward_backbone(self.backbone_t, img)[-1]
        else:
            F_t2s = self.forward_backbone(self.backbone_s, img)
            F_t2t = self.forward_backbone(self.backbone_t, img)
        ## 2. forward neck
        F_t2s_dsk, F_t2t_dsk = self.dsk_neck(F_t2s, F_t2t)
        ## 3. forward decode_head
        P_t2s = self.forward_decode_head(self.decode_head_s, F_t2s_dsk)
        P_t2t = self.forward_decode_head(self.decode_head_t, F_t2t_dsk)
        out = (P_t2s + P_t2t) / 2
        out = resize(
            input=out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        return out

    def _decode_head_forward_test(self, x, img_metas):
        """Run forward function and calculate loss for decode head in
        inference."""
        seg_logits = self.decode_head_s.forward_test(x, img_metas, self.test_cfg)
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
    
    def forward_discriminator(self, discriminator, seg_pred):
        dis_pred = discriminator(seg_pred)
        return dis_pred

    def forward_train(self, img, B_img):
        pass
        """Forward function for training."""

    def _get_segmentor_loss(self, decode_head, pred, gt_semantic_seg):
        losses = dict()
        loss_seg = decode_head.losses(pred, gt_semantic_seg)
        losses.update(loss_seg)
        loss_seg, log_vars_seg = self._parse_losses(losses)
        return loss_seg, log_vars_seg
    
    ## added by LYU: 2022/04/06
    def _get_gan_loss(self, discriminator, pred, domain, target_is_real):
        losses = dict()
        losses[f'loss_gan_{domain}'] = discriminator.gan_loss(pred, target_is_real)
        loss_dis, log_vars_dis = self._parse_losses(losses)
        ## added by LYU: 2022/04/06 only support one gan_loss
        ## auxiliary_ganloss: TBD
        return loss_dis, log_vars_dis
    
    ## added by LYU: 2022/05/11
    def _get_KD_loss(self, teacher, student, pred_name, T=3):
        losses = dict()
        losses[f'loss_KD_{pred_name}'] = self.KL_loss(teacher, student, T)
        loss_KD, log_vars_KD = self._parse_losses(losses)
        return loss_KD, log_vars_KD

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

        ## 1. towards all optimizers, clear gradients
        optimizer['backbone_s'].zero_grad()
        optimizer['backbone_t'].zero_grad()
        optimizer['decode_head_s'].zero_grad()
        optimizer['decode_head_t'].zero_grad()
        optimizer['discriminator_s'].zero_grad()
        optimizer['discriminator_t'].zero_grad()
        optimizer['discriminator_fs'].zero_grad()
        optimizer['discriminator_ft'].zero_grad()
        optimizer['dsk_neck'].zero_grad()

        self.set_requires_grad(self.backbone_s, False)
        self.set_requires_grad(self.backbone_t, False)
        self.set_requires_grad(self.decode_head_s, False)
        self.set_requires_grad(self.decode_head_t, False)
        self.set_requires_grad(self.discriminator_s, False)
        self.set_requires_grad(self.discriminator_t, False)
        self.set_requires_grad(self.discriminator_fs, False)
        self.set_requires_grad(self.discriminator_ft, False)
        self.set_requires_grad(self.dsk_neck, False)
        log_vars = dict()

        ## 1.1 forward backbone
        self.set_requires_grad(self.backbone_s, True)
        self.set_requires_grad(self.decode_head_s, True)
        self.set_requires_grad(self.backbone_t, True)
        self.set_requires_grad(self.decode_head_t, True)
        F_s2s_all = self.forward_backbone(self.backbone_s, data_batch['img'])
        F_t2s_all = self.forward_backbone(self.backbone_s, data_batch['B_img'])
        F_s2t_all = self.forward_backbone(self.backbone_t, data_batch['img'])
        F_t2t_all = self.forward_backbone(self.backbone_t, data_batch['B_img'])
        
        if isinstance(self.dsk_neck.in_channels, int):
            F_s2s = F_s2s_all[-1]
            F_t2s = F_t2s_all[-1]
            F_s2t = F_s2t_all[-1]
            F_t2t = F_t2t_all[-1]
        else:
            F_s2s = F_s2s_all
            F_t2s = F_t2s_all
            F_s2t = F_s2t_all
            F_t2t = F_t2t_all

        ## 1.2 forward dsk_neck
        self.set_requires_grad(self.dsk_neck, True)
        F_s2s_dsk, F_s2t_dsk = self.dsk_neck(F_s2s, F_s2t)
        F_t2s_dsk, F_t2t_dsk = self.dsk_neck(F_t2s, F_t2t)

        ## 1.3 forward head
        P_s2s = self.forward_decode_head(self.decode_head_s, F_s2s_dsk)
        P_t2s = self.forward_decode_head(self.decode_head_s, F_t2s_dsk)
        P_s2t = self.forward_decode_head(self.decode_head_t, F_s2t_dsk)
        P_t2t = self.forward_decode_head(self.decode_head_t, F_t2t_dsk)
        loss_seg_s2s, log_vars_seg_s2s = self._get_segmentor_loss(self.decode_head_s, P_s2s, data_batch['gt_semantic_seg'])
        log_vars.update(log_vars_seg_s2s)
        loss_seg_s2t, log_vars_seg_s2t = self._get_segmentor_loss(self.decode_head_t, P_s2t, data_batch['gt_semantic_seg'])
        log_vars_seg_s2t['loss_ce_seg_s2t'] = log_vars_seg_s2t.pop('loss_ce')
        log_vars_seg_s2t['acc_seg_s2t'] = log_vars_seg_s2t.pop('acc_seg')
        log_vars_seg_s2t['loss_seg_s2t'] = log_vars_seg_s2t.pop('loss')
        log_vars.update(log_vars_seg_s2t)
        loss_seg = loss_seg_s2s + loss_seg_s2t

        ## 2.2 forward&backward Ds/Dt
        ## 2.2.1 generator/prediction alignment Ds
        F_s2s = F_s2s_all[-1]
        F_t2s = F_t2s_all[-1]
        F_s2s_dis_sm = self.sw_softmax(F_s2s)
        F_t2s_dis_sm = self.sw_softmax(F_t2s)
        F_s2s_dis_oup = self.forward_discriminator(self.discriminator_s, F_s2s_dis_sm)
        F_t2s_dis_oup = self.forward_discriminator(self.discriminator_s, F_t2s_dis_sm)
        F_t2s_dis_oup = resize(
            input=F_t2s_dis_oup,
            size=data_batch['B_img'].shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        F_s2s_dis_oup = resize(
            input=F_s2s_dis_oup,
            size=data_batch['img'].shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        loss_dis_s, log_vars_dis_s = self._get_gan_loss(self.discriminator_s, F_t2s_dis_oup, 'F_t2s_ds_seg', 1)
        log_vars.update(log_vars_dis_s)

        ## 2.2.2 generator/prediction alignment Dt
        F_s2t = F_s2t_all[-1]
        F_t2t = F_t2t_all[-1]
        F_s2t_dis_sm = self.sw_softmax(F_s2t)
        F_t2t_dis_sm = self.sw_softmax(F_t2t)
        F_s2t_dis_oup = self.forward_discriminator(self.discriminator_t, F_s2t_dis_sm)
        F_t2t_dis_oup = self.forward_discriminator(self.discriminator_t, F_t2t_dis_sm)
        F_t2t_dis_oup = resize(
            input=F_t2t_dis_oup,
            size=data_batch['B_img'].shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        F_s2t_dis_oup = resize(
            input=F_s2t_dis_oup,
            size=data_batch['img'].shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        loss_dis_t, log_vars_dis_t = self._get_gan_loss(self.discriminator_t, F_s2t_dis_oup, 'F_s2t_dt_seg', 1)
        log_vars.update(log_vars_dis_t)
        loss_adv = loss_dis_t + loss_dis_s
        loss_stage1 = loss_seg + loss_adv
        loss_stage1.backward()
        optimizer['backbone_s'].step()
        optimizer['backbone_t'].step()
        optimizer['decode_head_s'].step()
        optimizer['decode_head_t'].step()
        optimizer['dsk_neck'].step()
        self.set_requires_grad(self.backbone_s, False)
        self.set_requires_grad(self.backbone_t, False)
        self.set_requires_grad(self.decode_head_s, False)
        self.set_requires_grad(self.decode_head_t, False)
        self.set_requires_grad(self.dsk_neck, False)

        ## 2.2.5 discriminator_s
        self.set_requires_grad(self.discriminator_s, True)
        F_s2s_dis_detach = F_s2s_dis_sm.detach()
        F_t2s_dis_detach = F_t2s_dis_sm.detach()
        F_s2s_dis_detach_oup = self.forward_discriminator(self.discriminator_s, F_s2s_dis_detach)
        F_t2s_dis_detach_oup = self.forward_discriminator(self.discriminator_s, F_t2s_dis_detach)
        F_t2s_dis_detach_oup = resize(
            input=F_t2s_dis_detach_oup,
            size=data_batch['B_img'].shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        F_s2s_dis_detach_oup = resize(
            input=F_s2s_dis_detach_oup,
            size=data_batch['img'].shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        loss_adv_s2s_ds, log_vars_adv_s2s_ds = self._get_gan_loss(self.discriminator_s, F_s2s_dis_detach_oup, 'F_s2s_ds', 1)
        loss_adv_s2s_ds.backward()
        log_vars.update(log_vars_adv_s2s_ds)
        loss_adv_t2s_ds, log_vars_adv_t2s_ds = self._get_gan_loss(self.discriminator_s, F_t2s_dis_detach_oup, 'F_t2s_ds', 0)
        loss_adv_t2s_ds.backward()
        log_vars.update(log_vars_adv_t2s_ds)
        optimizer['discriminator_s'].step()
        self.set_requires_grad(self.discriminator_s, False)

        ## 2.2.6 discriminator_t
        self.set_requires_grad(self.discriminator_t, True)
        F_s2t_dis_detach = F_s2t_dis_sm.detach()
        F_t2t_dis_detach = F_t2t_dis_sm.detach()
        F_s2t_dis_detach_oup = self.forward_discriminator(self.discriminator_t, F_s2t_dis_detach)
        F_t2t_dis_detach_oup = self.forward_discriminator(self.discriminator_t, F_t2t_dis_detach)
        F_t2t_dis_detach_oup = resize(
            input=F_t2t_dis_detach_oup,
            size=data_batch['B_img'].shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        F_s2t_dis_detach_oup = resize(
            input=F_s2t_dis_detach_oup,
            size=data_batch['img'].shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        loss_adv_t2t_dt, log_vars_adv_t2t_dt = self._get_gan_loss(self.discriminator_t, F_t2t_dis_detach_oup, 'F_t2t_dt', 1)
        loss_adv_t2t_dt.backward()
        log_vars.update(log_vars_adv_t2t_dt)
        loss_adv_s2t_dt, log_vars_adv_s2t_dt = self._get_gan_loss(self.discriminator_t, F_s2t_dis_detach_oup, 'F_s2t_dt', 0)
        loss_adv_s2t_dt.backward()
        log_vars.update(log_vars_adv_s2t_dt)
        optimizer['discriminator_t'].step()
        self.set_requires_grad(self.discriminator_t, False)
       
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
    
    ## added by LYU: 2022/05/12
    def MSE_loss(self, teacher, student):
        MSE_loss = nn.MSELoss()
        t = self.sw_softmax(teacher)
        s = self.sw_softmax(student)
        KD_loss = MSE_loss(s, t)
        return KD_loss

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
    
    ## added by LYU: 2022/05/11
    @staticmethod
    def KL_loss(teacher, student, T=5):
        KL_loss = nn.KLDivLoss(reduction='mean')(F.log_softmax(student/T, dim=1),
                             F.softmax(teacher/T, dim=1)) * (T * T)
        return KL_loss
