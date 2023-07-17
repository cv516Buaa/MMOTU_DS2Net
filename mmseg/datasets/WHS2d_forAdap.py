# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import numpy as np

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class WHS2dDataset_forAdap(CustomDataset):
    """WHS 2d CT/MR dataset for Domain adaptation.

    Args:
        split (str): Split txt file for domain A of MMOTU dataset .
    """
    CLASSES = ('background', 'LV', 'Myo', 'RV')
    
    PALETTE = [[0, 0, 0], [64, 0, 0], [0, 64, 0], [0, 0, 64]]


    def __init__(self, 
                 split, 
                 B_split=None, 
                 B_img_dir=None, 
                 B_img_suffix='.PNG',
                 B_ann_dir=None,
                 B_seg_map_suffix='_index.PNG', **kwargs):
        super(WHS2dDataset_forAdap, self).__init__(
            img_suffix='.PNG', seg_map_suffix='_index.PNG', split=split, **kwargs)
        assert osp.exists(self.img_dir) and self.split is not None

        self.B_img_dir = B_img_dir
        self.B_img_suffix = B_img_suffix
        self.B_ann_dir = B_ann_dir
        self.B_seg_map_suffix = B_seg_map_suffix
        self.B_split = B_split

        # join paths if data_root is specified
        if self.B_img_dir is not None:
            if not osp.isabs(self.B_img_dir):
                self.B_img_dir = osp.join(self.data_root, self.B_img_dir)
            if not (self.B_ann_dir is None or osp.isabs(self.B_ann_dir)):
                self.B_ann_dir = osp.join(self.data_root, self.B_ann_dir)
            if not (self.B_split is None or osp.isabs(self.B_split)):
                self.B_split = osp.join(self.data_root, self.B_split)
            # load annotations
            self.B_img_infos = self.load_annotations(self.B_img_dir, self.B_img_suffix,
                                               self.B_ann_dir,
                                               self.B_seg_map_suffix, self.B_split)
        else:
            self.B_img_infos = None

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.
        Args:
            idx (int): Index of data.
        Returns:
            dict: Training data and annotation after pipeline with new keys
                introduced by pipeline.
        """
        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx)
        assert len(self.B_img_infos) > 0
        idx_b = np.random.randint(0, len(self.B_img_infos))
        B_img_info = self.B_img_infos[idx_b]
        results = dict(img_info=img_info, ann_info=ann_info, B_img_info=B_img_info)
        self.pre_pipeline(results)
        return self.pipeline(results)
        
    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['seg_fields'] = []
        results['img_prefix'] = self.img_dir
        results['seg_prefix'] = self.ann_dir
        if not self.test_mode:
            results['B_img_prefix'] = self.B_img_dir
        if self.custom_classes:
            results['label_map'] = self.label_map