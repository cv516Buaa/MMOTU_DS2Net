_base_ = [
    '../../../configs/_base_/models/segformer_mit-b0.py',
    '../../../configs/_base_/datasets/whs_adapseg.py', '../../../configs/_base_/default_runtime.py',
]
model = dict(
    type='EncoderDecoder_forAdap',
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint='./pretrained/mit_b5.pth'),
        embed_dims=64,
        num_layers=[3, 6, 40, 3]),
    decode_head=dict(
        in_channels=[64, 128, 320, 512],
        num_classes=2
    ),
    discriminator=dict(
        type='AdapSegDiscriminator',
        gan_loss=dict(
            type='GANLoss',
            gan_type='vanilla',
            real_label_val=1.0,
            fake_label_val=0.0,
            loss_weight=0.005),
        norm_cfg=None,
        in_channels=2),
    test_cfg=dict(mode='slide', crop_size=(1024, 1024), stride=(768, 768)))

data = dict(samples_per_gpu=1, workers_per_gpu=1)

work_dir = './experiments/AdapSeg_segformerb5_256x256_40k_WHS/results/'

# learning policy
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1600,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)

total_iters = 40000
checkpoint_config = dict(by_epoch=False, interval=800)
evaluation = dict(interval=800, metric='mIoU', pre_eval=True)

# optimizer setting
optimizer = dict(
    backbone=dict(
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
    decode_head=dict(
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
    discriminators=dict(type='Adam', lr=0.00001, betas=(0.9, 0.99)))

runner = None
#use_ddp_wrapper = True
find_unused_parameters = True
