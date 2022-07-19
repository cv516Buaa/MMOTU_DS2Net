_base_ = [
    '../../../configs/_base_/models/segformer_mit-b0.py',
    '../../../configs/_base_/datasets/mmotu.py', '../../../configs/_base_/default_runtime.py',
    '../../../configs/_base_/schedules/schedule_80k.py'
]
model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint='./pretrained/mit_b0.pth')),
    decode_head=dict(
        num_classes=2,
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
work_dir = './experiments/segformerb0_769x769_80k_MMOTU/results/'
workflow = [('train', 1), ('val', 1)]
