import os.path

_base_ = [
    '../../../_base_/default_runtime.py',
    # 删掉了 coco.py，因为你自己完整定义了 dataloader，不需要继承它的配置，避免冲突
]

# ==============================================================================
# 0. 全局变量 (统一管理路径，方便修改)
# ==============================================================================
DATASET_TYPE = 'LDProsDataset'
DATA_ROOT = '/DATA/propose'
DATA_MODE = 'topdown'

# 确保这里的路径是对的
TRAIN_ANN = os.path.join(DATA_ROOT, 'train_final/merged_train_total_coco.json')
VAL_ANN =   os.path.join(DATA_ROOT, 'test_final/test_final.json')
TEST_ANN =  os.path.join(DATA_ROOT, 'test_final/test_final.json')
randomness = dict(seed=42, deterministic=False)
# ==============================================================================
# 1. Custom Imports (必须导入 Dataset!)
# ==============================================================================
custom_imports = dict(
    imports=[
        'mmpose.evaluation.metrics.prosthetics_metrics_baseline',
        'mmpose.models.heads.anatomy_aware_head',
        'mmpose.datasets.datasets.custom.ld_pros_dataset',
        'mmpose.evaluation.metrics.prosthetics_only_oks_metrics_baseline'
    ],
    allow_failed_imports=False
)

# ==============================================================================
# 2. Decoder Config
# ==============================================================================
codec = dict(
    type='SimCCLabel',
    input_size=(192, 256),
    sigma=(4.9, 5.66),
    simcc_split_ratio=2.0,
    normalize=False,
    use_dark=False
)

# ==============================================================================
# 3. Model Configuration
# ==============================================================================

max_epochs = 50
base_lr = 4e-3
train_batch_size = 64
val_batch_size = 32
train_cfg = dict(max_epochs=max_epochs, val_interval=10)
randomness = dict(seed=21)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=4e-3, weight_decay=0.05),
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0e-5,
        by_epoch=False,
        begin=0,
        end=1000),
    dict(
        # use cosine lr from 150 to 300 epoch
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=max_epochs // 2,
        end=max_epochs,
        T_max=max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True),
]

auto_scale_lr = dict(base_batch_size=512)


# ==============================================================================
# 3. Model Configuration (DWPose / RTMPose-l)
# ==============================================================================
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        _scope_='mmdet',
        type='CSPNeXt',
        arch='P5',
        expand_ratio=0.5,
        deepen_factor=1.,
        widen_factor=1.,
        out_indices=(4, ),
        channel_attention=True,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU'),
        init_cfg=dict(
            type='Pretrained',
            prefix='backbone.',
            checkpoint='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_simcc-ucoco_dw-ucoco_270e-256x192-4d6dfc62_20230728.pth'  # noqa: E501
        )),
    head=dict(
        type='RTMCCHead',
        in_channels=1024,
        out_channels=31,
        input_size=codec['input_size'],
        in_featuremap_size=(6, 8),
        simcc_split_ratio=codec['simcc_split_ratio'],
        final_layer_kernel_size=7,
        gau_cfg=dict(
            hidden_dims=256,
            s=128,
            expansion_factor=2,
            dropout_rate=0.,
            drop_path=0.,
            act_fn='SiLU',
            use_rel_bias=False,
            pos_enc=False),
        loss=dict(
            type='KLDiscretLoss',
            use_target_weight=True,
            beta=10.,
            label_softmax=True),
        decoder=codec),
    test_cfg=dict(flip_test=True, ))


# ==============================================================================
# 4. Data Pipeline
# ==============================================================================
train_pipeline = [
    dict(type='LoadImage', imdecode_backend='pillow'),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(type='ClampScale'),
    dict(type='RandomBBoxTransform'),
    # use_udp 建议先关掉，除非你明确知道你在做什么
    dict(type='TopdownAffine', input_size=codec['input_size'], use_udp=False),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]

val_pipeline = [
    dict(type='LoadImage', imdecode_backend='pillow'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size'], use_udp=False),
    dict(type='PackPoseInputs')
]

# ==============================================================================
# 5. Dataloaders (核心修正处)
# ==============================================================================
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

# ==============================================================================
# 6. Evaluators
# ==============================================================================
val_evaluator = dict(
    type='ProstheticsOKSMetric',
    ann_file=VAL_ANN,
)

test_evaluator = val_evaluator

# default_hooks = dict(
#     timer=dict(type='IterTimerHook'),
#     logger=dict(type='LoggerHook', interval=50),
#     param_scheduler=dict(type='ParamSchedulerHook'),
#     checkpoint=dict(type='CheckpointHook', interval=10),  # 每10轮保存一次模型
#     sampler_seed=dict(type='DistSamplerSeedHook'),  # 这个保留，用来设定随机种子
#
#     # 你的可视化配置
#     visualization=dict(
#         type='PoseVisualizationHook',
#         enable=True,
#         interval=1,
#         show=False,
#         # 强制使用绝对路径，确保你能找到图片
#         out_dir='/home/sora/workspace/mmpose/debug_output_force'
#     ),
# )