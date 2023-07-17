_base_ = [
    '../../../configs/_base_/models/segformer_mit-b0.py',
    '../../../configs/_base_/datasets/whs.py', '../../../configs/_base_/default_runtime.py',
    '../../../configs/_base_/schedules/schedule_40k.py'
]
model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint='./pretrained/mit_b5.pth'),
        embed_dims=64,
        num_layers=[3, 6, 40, 3]),
    decode_head=dict(
        in_channels=[64, 128, 320, 512],
        num_classes=4,
        sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=200000),
        loss_decode=[dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0, class_weight=[1.0, 2.5, 2.5, 2.5])]
    ),
    test_cfg=dict(mode='slide', crop_size=(1024, 1024), stride=(768, 768)))
# optimizer
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))

lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)
data = dict(samples_per_gpu=1, workers_per_gpu=1)

total_iters = 40000
checkpoint_config = dict(by_epoch=False, interval=800)
evaluation = dict(interval=800, metric='mDice', pre_eval=True)

work_dir = './experiments/DS2Net_segformerb5_256x256_40k_WHS/results/'
workflow = [('train', 1), ('val', 1)]
