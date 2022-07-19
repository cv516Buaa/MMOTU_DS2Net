_base_ = [
    '../../../configs/_base_/datasets/mmotu.py', '../../../configs/_base_/default_runtime.py',
    '../../../configs/_base_/schedules/schedule_80k.py'
]
# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (384, 384)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(448, 448), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
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
data = dict(
    train=dict(
        pipeline=train_pipeline),
    val=dict(
        pipeline=test_pipeline),
    test=dict(
        pipeline=test_pipeline))
model = dict(
    type='EncoderDecoder',
    pretrained='open-mmlab://resnet50_v1c',
    backbone=dict(
        type='ResNetV1c_vit',
        depth=50,
        num_stages=3,
        out_indices=(0, 1, 2),
        dilations=(1, 1, 2),
        strides=(1, 2, 2),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True,
        vit_input_indices=[2],
        vit_in_channels=1024,
        vit_embed_dims=768,
        vit_patch_size=1,
        vit_norm_cfg=dict(type='LN', eps=1e-6),
        vit_patch_norm=False,
        vit_feature_size=32,
        vit_with_cls_token=True,
        vit_drop_rate=0.0,
        vit_drop_path_rate=0.0,
        vit_num_layers=12,
        vit_mlp_ratio=4,
        vit_attn_drop_rate=0.0,
        vit_num_fcs=2,
        vit_qkv_bias=True,
        vit_act_cfg=dict(type='GELU'),
        vit_final_norm=False,
        vit_output_indices=-1),
    decode_head=dict(
        type='TransUnetHead',
        in_channels=[256, 512, 1024],
        in_index=[0, 1, 2],
        trans_in_channels=[768], 
        trans_in_index=[0], 
        trans_num_convs=(2, 2, 2),
        trans_upsample_flag=(False, True, True),
        upsample_rate=2,
        channels=256,
        num_classes=2,
        norm_cfg=norm_cfg,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
work_dir = './experiments/transunet_r50-vit_512x512_80k_MMOTU/results/'
workflow = [('train', 1), ('val', 1)]
