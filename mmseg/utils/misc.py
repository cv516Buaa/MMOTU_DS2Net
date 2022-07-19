# Copyright (c) OpenMMLab. All rights reserved.
import glob
import os.path as osp
import warnings

#####
# added by LYU: 2022/02/23
import mmcv
from mmseg.utils import get_root_logger
import numpy as np
import cv2

def find_latest_checkpoint(path, suffix='pth'):
    """This function is for finding the latest checkpoint.

    It will be used when automatically resume, modified from
    https://github.com/open-mmlab/mmdetection/blob/dev-v2.20.0/mmdet/utils/misc.py

    Args:
        path (str): The path to find checkpoints.
        suffix (str): File extension for the checkpoint. Defaults to pth.

    Returns:
        latest_path(str | None): File path of the latest checkpoint.
    """
    if not osp.exists(path):
        warnings.warn("The path of the checkpoints doesn't exist.")
        return None
    if osp.exists(osp.join(path, f'latest.{suffix}')):
        return osp.join(path, f'latest.{suffix}')

    checkpoints = glob.glob(osp.join(path, f'*.{suffix}'))
    if len(checkpoints) == 0:
        warnings.warn('The are no checkpoints in the path')
        return None
    latest = -1
    latest_path = ''
    for checkpoint in checkpoints:
        if len(checkpoint) < len(latest_path):
            continue
        # `count` is iteration number, as checkpoints are saved as
        # 'iter_xx.pth' or 'epoch_xx.pth' and xx is iteration number.
        count = int(osp.basename(checkpoint).split('_')[-1].split('.')[0])
        if count > latest:
            latest = count
            latest_path = checkpoint
    return latest_path


#####
# added by LYU: 2022/02/22 visulization of dataloader
def dataloader_visulization(cfg, dataloaders, vis_iters=10):
    # if workflow == 1, only visulize trainloader; 
    # if workflow == 2, visulize trainloader and valloader
    # Default: visualize the first 10 batches, each batch visulizes the first image of mini-batch
    # SUPPORT: cityscapes
    logger = get_root_logger(cfg.log_level)

    save_dir = cfg.work_dir + 'vis_results/'
    mmcv.mkdir_or_exist(osp.abspath(save_dir))
    logger.info("Save dataloader visulization to " + save_dir)

    for i in range(len(cfg.workflow)):
        for batch_id, data_batch in enumerate(dataloaders[i]):
            if batch_id >= 10:
                break         
            #img_name_tmp = data_batch['img_metas'].data[0][0]['filename']
            #img_tmp = cv2.imread(img_name_tmp)
            img_tmp = data_batch['B_img'].data[0][0].numpy().transpose(1, 2, 0) * 255
            img_gt_tmp = data_batch['gt_semantic_seg'].data[0][0].numpy().transpose(1, 2, 0)

            r = np.zeros((img_gt_tmp.shape[0], img_gt_tmp.shape[1], 1))
            g = np.zeros((img_gt_tmp.shape[0], img_gt_tmp.shape[1], 1))
            b = np.zeros((img_gt_tmp.shape[0], img_gt_tmp.shape[1], 1))
            for l in range(0, len(dataloaders[i].dataset.PALETTE)):
                r[img_gt_tmp == l] = dataloaders[i].dataset.PALETTE[l][0]
                g[img_gt_tmp == l] = dataloaders[i].dataset.PALETTE[l][1]
                b[img_gt_tmp == l] = dataloaders[i].dataset.PALETTE[l][2]

            gt_mask_tmp = np.zeros((img_gt_tmp.shape[0], img_gt_tmp.shape[1], 3))
            gt_mask_tmp[:, :, 0] = r.squeeze()
            gt_mask_tmp[:, :, 1] = g.squeeze()
            gt_mask_tmp[:, :, 2] = b.squeeze()
            outurl_img_tmp = save_dir + 'img' + str(i) + '_batchid' + str(batch_id) + '.png'
            outurl_gt_tmp = save_dir + 'gt' + str(i) + '_batchid' + str(batch_id) + '.png'
            cv2.imwrite(outurl_img_tmp, img_tmp)
            cv2.imwrite(outurl_gt_tmp, gt_mask_tmp)  
#####