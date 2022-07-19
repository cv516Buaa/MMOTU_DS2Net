# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class MMOTUDataset(CustomDataset):
    """MMOTU dataset.

    Args:
        split (str): Split txt file for MMOTU.
    """
    '''
    CLASSES = ('Chocolate_Cyst', 'SerousCystadenoma', 'Teratoma', 'ThecaCellTumor', 'SimpleCyst',
               'NormalOvary', 'MucinousCystadenoma', 'HighGradeSerousCystadenoma')
    
    PALETTE = [[64, 0, 0], [0, 64, 0], [0, 0, 64], [64, 0, 64],
               [64, 64, 0], [64, 64, 64], [0, 128, 0], [0, 0, 128]]
    '''
    CLASSES = ('background', 'OvarianTumor')
    
    PALETTE = [[0, 0, 0], [128, 0, 0]]

    def __init__(self, split, **kwargs):
        super(MMOTUDataset, self).__init__(
            img_suffix='.JPG', seg_map_suffix='_binary.PNG', split=split, **kwargs)
        assert osp.exists(self.img_dir) and self.split is not None
