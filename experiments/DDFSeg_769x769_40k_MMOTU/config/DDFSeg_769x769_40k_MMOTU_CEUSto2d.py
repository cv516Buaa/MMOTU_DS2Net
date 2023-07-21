_base_ = [
    '../../../configs/_base_/datasets/mmotu_adapseg.py', '../../../configs/_base_/default_runtime.py',
]
# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder_forDDFSeg',
    pretrained='torchvision://resnet18',
    backbone_s=dict(
        type='ResNetV1c',
        depth=18,
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
        depth=18,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    decode_head_s=dict(
        type='DAHead',
        in_channels=512,
        in_index=3,
        channels=128,
        pam_channels=64,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    decode_head_t=dict(
        type='DAHead',
        in_channels=512,
        in_index=3,
        channels=128,
        pam_channels=64,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    trans_head_s=dict(
        type='FCNDSFN_head',
        in_channels=512,
        in_index=3,
        channels=128,
        num_convs=3,
        dropout_ratio=0.1,
        num_classes=3,
        norm_cfg=dict(type='IN'),
        conv_cfg=dict(type='deconv', output_padding=1),
        align_corners=False),
    trans_head_t=dict(
        type='FCNDSFN_head',
        in_channels=512,
        in_index=3,
        channels=128,
        num_convs=3,
        dropout_ratio=0.1,
        num_classes=3,
        norm_cfg=dict(type='IN'),
        conv_cfg=dict(type='deconv', output_padding=1),
        align_corners=False),
    discriminator_s=dict(
        type='AdapSegDiscriminator',
        num_conv=2,
        gan_loss=dict(
            type='GANLoss',
            gan_type='vanilla',
            real_label_val=1.0,
            fake_label_val=0.0,
            loss_weight=0.005),
        norm_cfg=None,
        in_channels=2),
    discriminator_t=dict(
        type='AdapSegDiscriminator',
        num_conv=2,
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
    test_cfg=dict(mode='slide', crop_size=(1024, 1024), stride=(768, 768)))

work_dir = './experiments/DDFSeg_769x769_40k_MMOTU/results/'

# learning policy
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=800,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)

total_iters = 40000
checkpoint_config = dict(by_epoch=False, interval=800)
evaluation = dict(interval=800, metric='mIoU', pre_eval=True)


# optimizer setting
optimizer = dict(
    backbone_s=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005),
    backbone_t=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005),
    decode_head_s=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0005),
    decode_head_t=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0005),
    trans_head_s=dict(type='Adam', lr=0.00025, betas=(0.9, 0.99)),
    trans_head_t=dict(type='Adam', lr=0.00025, betas=(0.9, 0.99)),
    discriminator_s=dict(type='Adam', lr=0.00025, betas=(0.9, 0.99)),
    discriminator_t=dict(type='Adam', lr=0.00025, betas=(0.9, 0.99))
    )

'''
# optimizer setting
optimizer = dict(
    backbone_s=dict(
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        })),
    backbone_t=dict(
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        })),
    decode_head_s=dict(
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        })),
    decode_head_t=dict(
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        })),
    trans_head_s=dict(
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01),
    trans_head_t=dict(
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01),
    discriminator_s=dict(type='Adam', lr=0.00001, betas=(0.9, 0.99)),
    discriminator_t=dict(type='Adam', lr=0.00001, betas=(0.9, 0.99)))
'''
runner = None
#use_ddp_wrapper = True
find_unused_parameters = True


data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        img_dir='OTU_3d/images',
        ann_dir='OTU_3d/annotations',
        split='OTU_3d/train.txt',
        B_img_dir = 'OTU_2d/images',
        #B_ann_dir='OTU_3d/annotations', # unsupervised domain adaptation -> no annotations of domain B
        B_split = 'OTU_2d/train.txt'),
    # target domain for validation
    val=dict(
        img_dir='OTU_2d/images',
        ann_dir='OTU_2d/annotations', 
        split='OTU_2d/val.txt'),
    test=dict(
        img_dir='OTU_2d/images',
        ann_dir='OTU_2d/annotations',
        split='OTU_2d/val.txt'))
