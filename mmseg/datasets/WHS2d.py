# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import numpy as np

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class WHS2dDataset(CustomDataset):
    """WHS 2d CT/MR dataset for Semantic Segmentation.

    Args:
        split (str): Split txt file for domain A of MMOTU dataset .
    """
    CLASSES = ('background', 'LV', 'Myo', 'RV')
    
    PALETTE = [[0, 0, 0], [64, 0, 0], [0, 64, 0], [0, 0, 64]]

    def __init__(self, split, **kwargs):
        super(WHS2dDataset, self).__init__(
            img_suffix='.PNG', seg_map_suffix='_index.PNG', split=split, **kwargs)
        assert osp.exists(self.img_dir) and self.split is not None