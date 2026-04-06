import os.path


_base_ = [
    '../_base_/default_runtime.py',
]

DATASET_TYPE = 'AmputationCOCO'
DATA_ROOT = './data'
DATA_MODE = 'topdown'

TRAIN_ANN = 'annotations/person_keypoints_train2014.json'
VAL_ANN =   'annotations/person_keypoints_val2014.json'
TEST_ANN =  'annotations/person_keypoints_val2014.json'
randomness = dict(seed=42, deterministic=False)

amputation_types = [
    {'name': 'Above Left Elbow', 'limb_id': 'left_arm', 'p': 5, 'd': 7, 'r': 17, 'lost_pts': [7, 9],
     'dp_parts': [10]},
    {'name': 'Below Left Elbow', 'limb_id': 'left_arm', 'p': 7, 'd': 9, 'r': 19, 'lost_pts': [9],
     'dp_parts': [12]},
    {'name': 'Above Right Elbow', 'limb_id': 'right_arm', 'p': 6, 'd': 8, 'r': 18, 'lost_pts': [8, 10],
     'dp_parts': [11]},
    {'name': 'Below Right Elbow', 'limb_id': 'right_arm', 'p': 8, 'd': 10, 'r': 20, 'lost_pts': [10],
     'dp_parts': [13]},
    {'name': 'Above Left Knee', 'limb_id': 'left_leg', 'p': 11, 'd': 13, 'r': 21, 'lost_pts': [13, 15],
     'dp_parts': [7]},
    {'name': 'Below Left Knee', 'limb_id': 'left_leg', 'p': 13, 'd': 15, 'r': 23, 'lost_pts': [15],
     'dp_parts': [9]},
    {'name': 'Above Right Knee', 'limb_id': 'right_leg', 'p': 12, 'd': 14, 'r': 22, 'lost_pts': [14, 16],
     'dp_parts': [6]},
    {'name': 'Below Right Knee', 'limb_id': 'right_leg', 'p': 14, 'd': 16, 'r': 24, 'lost_pts': [16],
     'dp_parts': [8]},
]

custom_imports = dict(
    imports=[
        'mmpose.datasets.datasets.custom.amputation_coco',
        'mmpose.evaluation.metrics.distal_metric',
    ],
    allow_failed_imports=False
)

input_codec = dict(
    type='UDPHeatmap',
    input_size=(192, 256),
    heatmap_size=(192, 256),
    sigma=4.0,
)

target_codec = dict(
    type='UDPHeatmap',
    input_size=(192, 256),
    heatmap_size=(48, 64),
    sigma=2.0,
)

optim_wrapper = dict(
    optimizer=dict(
        type='AdamW',
        lr=5e-5,
        betas=(0.9, 0.999),
        weight_decay=0.1),
    paramwise_cfg=dict(
        num_layers=24,
        layer_decay_rate=0.8,
        custom_keys={
            'bias': dict(decay_mult=0.0),
            'pos_embed': dict(decay_mult=0.0),
            'relative_position_bias_table': dict(decay_mult=0.0),
            'norm': dict(decay_mult=0.0),
        },
    ),
    constructor='LayerDecayOptimWrapperConstructor',
    clip_grad=dict(max_norm=1., norm_type=2),
)

param_scheduler = [
    dict(type='LinearLR', begin=0, end=100, start_factor=0.001, by_epoch=False),
    dict(type='CosineAnnealingLR', T_max=10, by_epoch=True)
]

train_cfg = dict(
    by_epoch=True,
    max_epochs=100,
    val_interval=10
)


model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=None,
        std=None,
        bgr_to_rgb=False
    ),
    backbone=dict(
        type='mmpretrain.VisionTransformer',
        arch='large',
        in_channels=25,
        img_size=(256, 192),
        patch_size=16,
        qkv_bias=True,
        drop_path_rate=0.5,
        with_cls_token=False,
        out_type='featmap',
        patch_cfg=dict(padding=2),
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmpose/v1/pretrained_models/mae_pretrain_vit_large_20230913.pth'),
    ),
    head=dict(
        type='HeatmapHead',
        in_channels=1024,
        out_channels=8,
        deconv_out_channels=(256, 256),
        deconv_kernel_sizes=(4, 4),
        loss=dict(
            type='KeypointMSELoss',
            use_target_weight=True
        ),
        decoder=target_codec,
    ),
    test_cfg=dict(
        flip_test=False,
        shift_heatmap=False,
    )
)


train_pipeline = [
    dict(type='LoadImage', imdecode_backend='pillow'),
    dict(type='GetBBoxCenterScale'),
    dict(
        type='SimulateAmputationMath',
        amputation_types=amputation_types,
        prob=0.5,
        is_train=True),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomBBoxTransform'),
    dict(type='TopdownAffine', input_size=target_codec['input_size'], use_udp=True),
    dict(type='GenerateAmputationHeatmaps',
         input_codec_cfg=input_codec,
         target_codec_cfg=target_codec,
         amputation_types=amputation_types),
    dict(type='PackPoseInputs')
]

val_pipeline = [
    dict(type='LoadImage', imdecode_backend='pillow'),
    dict(type='GetBBoxCenterScale'),
    dict(
        type='SimulateAmputationMath',
        amputation_types=amputation_types,
        prob=0.5,
        is_train=False),
    dict(type='TopdownAffine', input_size=target_codec['input_size'], use_udp=True),
    dict(type='GenerateAmputationHeatmaps',
         input_codec_cfg=input_codec,
         target_codec_cfg=target_codec,
         amputation_types=amputation_types),
    dict(type='PackPoseInputs')
]


train_dataloader = dict(
    batch_size=64,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=DATASET_TYPE,
        data_root=DATA_ROOT,
        ann_file=TRAIN_ANN,
        data_prefix=dict(img='train2014/'),
        pipeline=train_pipeline,
    )
)

val_dataloader = dict(
    batch_size=32, num_workers=4, persistent_workers=True, drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=DATASET_TYPE, data_root=DATA_ROOT, ann_file=VAL_ANN,
        data_prefix=dict(img='val2014'), pipeline=val_pipeline, test_mode=True,
    )
)

test_dataloader = val_dataloader

val_evaluator = dict(
    type='DistalOKSMetric',
    amputation_types=amputation_types
)

test_evaluator = val_evaluator

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=10, max_keep_ckpts=5),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)

visualizer = dict(
    type='PoseLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(
            type='WandbVisBackend',
            init_kwargs=dict(
                project='distal-prediction',
                name='heatmap-input',
                entity='qitianye1104'
            )
        )
    ],
    name='visualizer'
)