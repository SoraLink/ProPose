import os

_base_ = ['../../../_base_/default_runtime.py']

DATASET_TYPE = 'LDProsYoloDataset'
DATA_ROOT = '/home/sora/workspace/dataset/pros_final'
DATA_MODE = 'bottomup'  # 🌟 YOLO 是 Bottom-up，这里必须改！

TRAIN_ANN = os.path.join(DATA_ROOT, 'train_final/train_final.json')
VAL_ANN = os.path.join(DATA_ROOT, 'test_final/test_final.json')
TEST_ANN = os.path.join(DATA_ROOT, 'test_final/test_final.json')
randomness = dict(seed=42, deterministic=False)

custom_imports = dict(
    imports=[
        'mmpose.evaluation.metrics.prosthetics_dekr_metrics_baseline',  # 你的 Metric 路径 (请确保文件名一致)
        'mmpose.models.heads.combined_dekr_anatomy_aware_head',  # 🌟 我们刚刚写的 YOLOX Head 路径
        'mmpose.datasets.datasets.custom.ld_pros_yolo_dataset',  # 你的数据集路径
    ],
    allow_failed_imports=False
)


# runtime
train_cfg = dict(max_epochs=140, val_interval=1)

# optimizer
optim_wrapper = dict(optimizer=dict(
    type='Adam',
    lr=1e-3,
))

# learning policy
param_scheduler = [
    dict(
        type='LinearLR', begin=0, end=500, start_factor=0.001,
        by_epoch=False),  # warm-up
    dict(
        type='MultiStepLR',
        begin=0,
        end=140,
        milestones=[90, 120],
        gamma=0.1,
        by_epoch=True)
]

# automatically scaling LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=80)


# codec settings
codec = dict(
    type='SPR',
    input_size=(512, 512),
    heatmap_size=(128, 128),
    sigma=(4, 2),
    minimal_diagonal_length=32**0.5,
    generate_keypoint_heatmaps=True,
    decode_max_instances=30)

# model settings
model = dict(
    type='BottomupPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type='HRNet',
        in_channels=3,
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4, ),
                num_channels=(64, )),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(32, 64)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(32, 64, 128)),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(32, 64, 128, 256),
                multiscale_output=True)),
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmpose/'
            'pretrain_models/hrnet_w32-36af842e.pth'),
    ),
    neck=dict(
        type='FeatureMapProcessor',
        concat=True,
    ),
    head=dict(
        type='CombinedDEKRAnatomyAwareHead',
        in_channels=480,
        num_keypoints=31,
        heatmap_loss=dict(type='KeypointMSELoss', use_target_weight=True),
        displacement_loss=dict(
            type='SoftWeightSmoothL1Loss',
            use_target_weight=True,
            supervise_empty=False,
            beta=1 / 9,
            loss_weight=0.002,
        ),
        type_loss_weight=0.0005,
        tau=0.2,
        bio_loss_weight=0.0003,
        with_contrastive=False,
        decoder=codec,
    ),
    test_cfg=dict(
        multiscale_test=False,
        nms_dist_thr=0.05,
        shift_heatmap=True,
        align_corners=False))

# enable DDP training when rescore net is used
find_unused_parameters = False

# base dataset settings
dataset_type = 'LDProsYoloDataset'
data_mode = 'bottomup'


# pipelines
train_pipeline = [
    dict(type='LoadImage', imdecode_backend='pillow'),
    dict(type='BottomupRandomAffine', input_size=codec['input_size']),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='BottomupGetHeatmapMask'),
    dict(type='PackPoseInputs'),
]
val_pipeline = [
    dict(type='LoadImage', imdecode_backend='pillow'),
    dict(
        type='BottomupResize',
        input_size=codec['input_size'],
        size_factor=32,
        resize_mode='expand'),
    dict(
        type='PackPoseInputs',
        meta_keys=('id', 'img_id', 'img_path', 'crowd_index', 'ori_shape',
                   'img_shape', 'input_size', 'input_center', 'input_scale',
                   'flip', 'flip_direction', 'flip_indices', 'raw_ann_info',
                   'skeleton_links'))
]

# data loaders
train_dataloader = dict(
    batch_size=8,  # 注意：如果你显存爆了，改小点，同时 lr 会被 auto_scale_lr 自动按比例缩小
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=DATASET_TYPE,
        data_root=DATA_ROOT,
        data_mode=DATA_MODE,
        serialize_data=False,
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        ann_file=TRAIN_ANN,
        data_prefix=dict(img='train_final/images/'),
        pipeline=train_pipeline,
    ))

val_dataloader = dict(
    batch_size=1,  # 验证时 batch_size 必须是 1 才能正确算指标
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=DATASET_TYPE,
        data_root=DATA_ROOT,
        data_mode=DATA_MODE,
        serialize_data=False,
        ann_file=VAL_ANN,
        data_prefix=dict(img='test_final/images/'),
        test_mode=True,
        pipeline=val_pipeline,
    ))
test_dataloader = val_dataloader

# evaluators
val_evaluator = dict(
    type='ProstheticsDEKRMetric',
    ann_file=VAL_ANN,
    nms_mode='none',
    score_mode='keypoint',
)
test_evaluator = val_evaluator

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,           # 每隔 1 个 epoch 保存一次（这样第 1 个 epoch 跑完就会存）
        max_keep_ckpts=3,     # ⚠️ 强烈建议加上这个！只保留最新的 3 个权重，防止硬盘被撑爆
        save_best='auto',     # 自动保存验证集上指标最好的权重
        rule='greater'        # 指标越大越好
    )
)