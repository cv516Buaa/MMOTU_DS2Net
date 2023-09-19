_base_ = [
    '../../../configs/_base_/datasets/whs_adapseg.py', '../../../configs/_base_/default_runtime.py',
]
# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder_forDAFormer',
    pretrained=None,
    backbone=dict(
        type='MixVisionTransformer',
        init_cfg=dict(type='Pretrained', checkpoint='./pretrained/mit_b5.pth'),
        in_channels=3,
        embed_dims=64,
        num_stages=4,
        num_layers=[3, 6, 40, 3],
        num_heads=[1, 2, 5, 8],
        patch_sizes=[7, 3, 3, 3],
        sr_ratios=[8, 4, 2, 1],
        out_indices=(0, 1, 2, 3),
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1),
    decode_head=dict(
        type='SegformerHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    cross_EMA = dict(
        ## two types: 'single_t', 'single_decoder'
        type='single_t',
        training_ratio=0.25,
        decay=0.999,
        pseudo_threshold=0.975,
        pseudo_rare_threshold=0.8,
        pseudo_class_weight=[1.01, 2.01],
        backbone_EMA=dict(
            type='MixVisionTransformer',
            init_cfg=dict(type='Pretrained', checkpoint='./pretrained/mit_b5.pth'),
            in_channels=3,
            embed_dims=64,
            num_stages=4,
            num_layers=[3, 6, 40, 3],
            num_heads=[1, 2, 5, 8],
            patch_sizes=[7, 3, 3, 3],
            sr_ratios=[8, 4, 2, 1],
            out_indices=(0, 1, 2, 3),
            mlp_ratio=4,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.1),
        decode_head_EMA=dict(
            type='SegformerHead',
            in_channels=[64, 128, 320, 512],
            in_index=[0, 1, 2, 3],
            channels=256,
            dropout_ratio=0.1,
            num_classes=2,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0, class_weight=[1.0, 2.0]))
    ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(1024, 1024), stride=(768, 768)))

work_dir = './experiments/DAFormer_256x256_40k_WHS/results/'

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
        })))

runner = None
#use_ddp_wrapper = True
find_unused_parameters = True


data = dict(
samples_per_gpu=1, 
workers_per_gpu=1,
train=dict(
    img_dir='CT_withGT/images',
    ann_dir='CT_withGT/annotations',
    split='CT_withGT/trainval.txt',
    B_img_dir = 'MR_withGT/images',
    B_split = 'MR_withGT/trainval.txt'),
# target domain for validation
val=dict(
    img_dir='MR_withGT/images',
    ann_dir='MR_withGT/annotations', 
    split='MR_withGT/val.txt',
    ),
test=dict(
    img_dir='MR_withGT/images',
    ann_dir='MR_withGT/annotations',
    split='MR_withGT/val.txt')
    )
