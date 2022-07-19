# Copyright (c) OpenMMLab. All rights reserved.
from .backbones import *  # noqa: F401,F403
from .builder import (BACKBONES, HEADS, LOSSES, SEGMENTORS, build_backbone,
                      build_head, build_loss, build_segmentor, build_discriminator)
from .decode_heads import *  # noqa: F401,F403
from .losses import *  # noqa: F401,F403
from .necks import *  # noqa: F401,F403
from .segmentors import *  # noqa: F401,F403

## added by LYU: 2022/03/28
from .discriminators import * # noqa: F401,F403

__all__ = [
    'BACKBONES', 'HEADS', 'LOSSES', 'SEGMENTORS', 'build_backbone',
    'build_head', 'build_loss', 'build_segmentor', 'build_discriminator'
]
