_base_ = [
    '../../../configs/_base_/models/fcn_unet_s5-d16.py',
    '../../../configs/_base_/datasets/mmotu.py', '../../../configs/_base_/default_runtime.py',
    '../../../configs/_base_/schedules/schedule_80k.py'
]
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(448, 448),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
model = dict(
    decode_head=dict(num_classes=2),
    auxiliary_head=dict(num_classes=2),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
data = dict(
    val=dict(
        pipeline=test_pipeline),)
work_dir = './experiments/unet_fcn_512x512_80k_MMOTU/results/'
workflow = [('train', 1), ('val', 1)]
