import os.path

_base_ = [
    '../../../_base_/default_runtime.py',
    # 删掉了 coco.py，因为你自己完整定义了 dataloader，不需要继承它的配置，避免冲突
]

# ==============================================================================
# 0. 全局变量 (统一管理路径，方便修改)
# ==============================================================================
DATASET_TYPE = 'LDProsDataset'
DATA_ROOT = '/home/sora/workspace/dataset/ldpose_final'
DATA_MODE = 'topdown'

# 确保这里的路径是对的
TRAIN_ANN = os.path.join(DATA_ROOT, 'pros_annotations/labels_train_final.json')
VAL_ANN =   os.path.join(DATA_ROOT, 'pros_annotations/labels_val_final.json')
TEST_ANN =  os.path.join(DATA_ROOT, 'pros_annotations/labels_test_final.json')
randomness = dict(seed=42, deterministic=False)
# ==============================================================================
# 1. Custom Imports (必须导入 Dataset!)
# ==============================================================================
custom_imports = dict(
    imports=[
        'mmpose.evaluation.metrics.prosthetics_metrics_baseline',
        'mmpose.models.heads.anatomy_aware_head',
        'mmpose.datasets.datasets.custom.ld_pros_dataset',
    ],
    allow_failed_imports=False
)

# ==============================================================================
# 2. Decoder Config
# ==============================================================================
codec = dict(
    type='UDPHeatmap',
    input_size=(192, 256),
    heatmap_size=(48, 64),
    sigma=2.0
)

# ==============================================================================
# 3. Model Configuration
# ==============================================================================

optim_wrapper = dict(
    optimizer=dict(
        type='AdamW', lr=5e-4, betas=(0.9, 0.999), weight_decay=0.05),
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
    dict(type='LinearLR', begin=0, end=500, start_factor=0.001, by_epoch=False),
    dict(type='CosineAnnealingLR', T_max=150, by_epoch=True)
]

train_cfg = dict(
    by_epoch=True,
    max_epochs=150,    # 训练多少轮 (建议 210 或 100)
    val_interval=5    # 每多少轮验证一次 (10 轮一次比较合适)
)

model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True
    ),
    backbone=dict(
        type='mmpretrain.VisionTransformer',
        arch='base',
        img_size=(256, 192),
        patch_size=16,
        qkv_bias=True,
        drop_path_rate=0.3,
        with_cls_token=False,
        out_type='featmap',
        patch_cfg=dict(padding=2),
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmpose/'
                       'v1/pretrained_models/mae_pretrain_vit_base_20230913.pth'),
    ),
    head=dict(
        type='AnatomyAwareHead',
        in_channels=768,
        out_channels=25, # 对应 Dataset 的 25 个点
        deconv_out_channels=(256, 256),
        deconv_kernel_sizes=(4, 4),
        loss=dict(
            type='KeypointMSELoss',
            use_target_weight=True
        ),
        decoder=codec,
        type_loss_weight=0.03
    ),
    test_cfg=dict(
        flip_mode='heatmap',
        shift_heatmap=False,
    )
)

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
    dict(type='TopdownAffine', input_size=codec['input_size'], use_udp=True),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]

val_pipeline = [
    dict(type='LoadImage', imdecode_backend='pillow'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size'], use_udp=True),
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
        data_prefix=dict(img='ldpose_train/'),
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
        data_prefix=dict(img='ldpose_val/'),
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
        data_prefix=dict(img='ldpose_test/'),
        pipeline=val_pipeline,
        test_mode=True,
    )
)

# ==============================================================================
# 6. Evaluators
# ==============================================================================
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
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=10),  # 每10轮保存一次模型
    sampler_seed=dict(type='DistSamplerSeedHook'),  # 这个保留，用来设定随机种子

    # 你的可视化配置
    visualization=dict(
        type='PoseVisualizationHook',
        enable=True,
        interval=1,
        show=False,
        # 强制使用绝对路径，确保你能找到图片
        out_dir='/home/sora/workspace/mmpose/debug_output_force'
    ),
)