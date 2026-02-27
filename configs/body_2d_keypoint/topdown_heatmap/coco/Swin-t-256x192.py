import os

_base_ = ['../../../_base_/default_runtime.py']

DATASET_TYPE = 'LDProsDataset'
DATA_ROOT = '/home/sora/workspace/dataset/pros_final'
DATA_MODE = 'topdown'

TRAIN_ANN = os.path.join(DATA_ROOT, 'train_final/train_final.json')
VAL_ANN =   os.path.join(DATA_ROOT, 'test_final/test_final.json')
TEST_ANN =  os.path.join(DATA_ROOT, 'test_final/test_final.json')

custom_imports = dict(
    imports=[
        'mmpose.evaluation.metrics.prosthetics_metrics_baseline',
        'mmpose.models.heads.class_balanced_anatomy_aware_head',
        'mmpose.datasets.datasets.custom.ld_pros_dataset',
    ],
    allow_failed_imports=False
)

# runtime
train_cfg = dict(max_epochs=50, val_interval=10)

# optimizer
optim_wrapper = dict(optimizer=dict(
    type='Adam',
    lr=5e-4,
))

# learning policy
param_scheduler = [
    dict(
        type='LinearLR', begin=0, end=500, start_factor=0.001,
        by_epoch=False),  # warm-up
    dict(
        type='MultiStepLR',
        begin=0,
        end=210,
        milestones=[170, 200],
        gamma=0.1,
        by_epoch=True)
]

# automatically scaling LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=64)

# hooks
default_hooks = dict(checkpoint=dict(save_best='coco/AP', rule='greater'))

# codec settings
codec = dict(
    type='MSRAHeatmap', input_size=(192, 256), heatmap_size=(48, 64), sigma=2)

# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type='SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(3, ),
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://github.com/SwinTransformer/storage/releases/'
            'download/v1.0.0/swin_tiny_patch4_window7_224.pth'),
    ),
    head=dict(
        type='HeatmapHead',
        in_channels=768,
        out_channels=31,
        loss=dict(type='KeypointMSELoss', use_target_weight=True),
        decoder=codec),
    test_cfg=dict(
        flip_test=True,
        flip_mode='heatmap',
        shift_heatmap=True,
    ))

# base dataset settings
dataset_type = 'LDProsDataset'
data_mode = 'topdown'

# pipelines
train_pipeline = [
    dict(type='LoadImage', imdecode_backend='pillow'),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(type='RandomBBoxTransform'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]

val_pipeline = [
    dict(type='LoadImage', imdecode_backend='pillow'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PackPoseInputs')
]

# data loaders
train_dataloader = dict(
    batch_size=64,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=DATASET_TYPE,
        data_root=DATA_ROOT,
        ann_file=TRAIN_ANN,
        data_prefix=dict(img='train_final/images/'),
        pipeline=train_pipeline,
    )
)
val_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=DATASET_TYPE, # <--- 修正
        data_root=DATA_ROOT,
        ann_file=VAL_ANN,
        data_prefix=dict(img='test_final/images/'),
        pipeline=val_pipeline,
        test_mode=True,
    )
)
test_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=DATASET_TYPE, # <--- 修正
        data_root=DATA_ROOT,
        ann_file=TEST_ANN,
        data_prefix=dict(img='test_final/images/'),
        pipeline=val_pipeline,
        test_mode=True,
    )
)

# evaluators
val_evaluator = dict(
    type='ProstheticsMetric', # 这里的 type 才是 Metric 的名字
    ann_file=VAL_ANN,         # 使用变量，保持一致
    score_thr=0.3,
)

test_evaluator = dict(
    type='ProstheticsMetric',
    ann_file=TEST_ANN,        # 使用变量
    score_thr=0.3,
)

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50), # 打印日志
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=5, max_keep_ckpts=-1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)

visualizer = dict(
    type='PoseLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),  # 保留本地日志记录
        dict(
            type='WandbVisBackend',    # 🌟 开启 W&B 魔法
            init_kwargs=dict(
                project='prosthetics-pose-estimation',  # W&B 上的项目名称
                name='Swin-t-prosthetics_CB_loss_256x192',         # 这次 Run 的名字
                entity='qitianye1104'                    # (可选) 你的 W&B 账号名或团队名
            )
        )
    ],
    name='visualizer'
)