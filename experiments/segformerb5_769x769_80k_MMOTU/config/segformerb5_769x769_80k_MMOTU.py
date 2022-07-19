_base_ = [
    '../../segformerb0_769x769_80k_MMOTU/config/segformerb0_769x769_80k_MMOTU.py',
]
model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint='./pretrained/mit_b5.pth'),
        embed_dims=64,
        num_layers=[3, 6, 40, 3]),
    decode_head=dict(
        in_channels=[64, 128, 320, 512],
    ),
    test_cfg=dict(mode='slide', crop_size=(1024, 1024), stride=(768, 768)))
work_dir = './experiments/segformerb5_769x769_80k_MMOTU/results/'
