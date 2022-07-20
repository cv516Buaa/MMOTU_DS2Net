_base_ = [
    '../../../configs/_base_/datasets/mmotu_adapseg.py', '../../../configs/_base_/default_runtime.py'
]

# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder_forDS2Net',
    pretrained='open-mmlab://resnet50_v1c',
    dsk_neck=dict(
        type='DSKNeck',
        in_channels=2048,
        r=8,
        L=32),
    backbone_s=dict(
        type='ResNetV1c',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    backbone_t=dict(
        type='ResNetV1c',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    decode_head_s=dict(
        type='PSPHead',
        in_channels=2048,
        #in_channels=2304,
        in_index=0,
        channels=512,
        pool_scales=(1, 2, 3, 6),
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    decode_head_t=dict(
        type='PSPHead',
        in_channels=2048,
        #in_channels=2304,
        in_index=0,
        channels=512,
        pool_scales=(1, 2, 3, 6),
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    discriminator_s=dict(
        type='AdapSegDiscriminator',
        gan_loss=dict(
            type='GANLoss',
            gan_type='vanilla',
            real_label_val=1.0,
            fake_label_val=0.0,
            loss_weight=0.005),
        norm_cfg=None,
        in_channels=2048),
    discriminator_t=dict(
        type='AdapSegDiscriminator',
        gan_loss=dict(
            type='GANLoss',
            gan_type='vanilla',
            real_label_val=1.0,
            fake_label_val=0.0,
            loss_weight=0.005),
        norm_cfg=None,
        in_channels=2048),
    discriminator_fs=dict(
        type='AdapSegDiscriminator',
        gan_loss=dict(
            type='GANLoss',
            gan_type='vanilla',
            real_label_val=1.0,
            fake_label_val=0.0,
            loss_weight=0.005),
        norm_cfg=None,
        in_channels=2),
    discriminator_ft=dict(
        type='AdapSegDiscriminator',
        gan_loss=dict(
            type='GANLoss',
            gan_type='vanilla',
            real_label_val=1.0,
            fake_label_val=0.0,
            loss_weight=0.005),
        norm_cfg=None,
        in_channels=2),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    )
work_dir = './experiments/DS2Net_pspnet_r50-d8_769x769_20k_MMOTU/results/'

# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=1e-5, by_epoch=False)

total_iters = 20000
checkpoint_config = dict(by_epoch=False, interval=400)
evaluation = dict(interval=400, metric='mIoU', pre_eval=True)

# optimizer setting
optimizer = dict(
    backbone_s=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005),
    backbone_t=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005),
    dsk_neck=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005),
    decode_head_s=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0005),
    decode_head_t=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0005),
    discriminator_s=dict(type='Adam', lr=0.00025, betas=(0.9, 0.99)),
    discriminator_t=dict(type='Adam', lr=0.00025, betas=(0.9, 0.99)),
    discriminator_fs=dict(type='Adam', lr=0.00025, betas=(0.9, 0.99)),
    discriminator_ft=dict(type='Adam', lr=0.00025, betas=(0.9, 0.99)),
    )

runner = None
#use_ddp_wrapper = True
find_unused_parameters = True
