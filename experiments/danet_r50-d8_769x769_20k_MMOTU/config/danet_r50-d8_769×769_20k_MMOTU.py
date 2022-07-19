_base_ = [
    '../../../configs/_base_/models/danet_r50-d8.py',
    '../../../configs/_base_/datasets/mmotu.py', '../../../configs/_base_/default_runtime.py',
    '../../../configs/_base_/schedules/schedule_20k.py'
]
model = dict(
    decode_head=dict(
        num_classes=2,
    ),
    auxiliary_head=dict(
        num_classes=2))
work_dir = './experiments/danet_r50-d8_769x769_20k_MMOTU/results/'
workflow = [('train', 1), ('val', 1)]
