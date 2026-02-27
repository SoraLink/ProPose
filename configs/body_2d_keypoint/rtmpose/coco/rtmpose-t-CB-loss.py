import os.path

_base_ = [
    '../../../_base_/default_runtime.py',
]

# ==============================================================================
# 1. 基础与路径配置
# ==============================================================================
DATASET_TYPE = 'LDProsDataset'
DATA_ROOT = '/home/sora/workspace/dataset/pros_final'
DATA_MODE = 'topdown'

TRAIN_ANN = os.path.join(DATA_ROOT, 'train_final/train_final.json')
VAL_ANN =   os.path.join(DATA_ROOT, 'test_final/test_final.json')
TEST_ANN =  os.path.join(DATA_ROOT, 'test_final/test_final.json')
randomness = dict(seed=42, deterministic=False)

custom_imports = dict(
    imports=[
        'mmpose.evaluation.metrics.prosthetics_metrics_baseline',
        # 🌟 指向你新写的 RTM Head 文件
        'mmpose.models.heads.combined_RTM_anatomy_aware_head',
        'mmpose.datasets.datasets.custom.ld_pros_dataset',
    ],
    allow_failed_imports=False
)

# 🌟 加载 RTMPose 在 COCO 上的预训练权重进行微调 (请根据你实际使用的模型体积替换链接)
# ==============================================================================
# 2. Decoder Config (RTMPose 使用 SimCC)
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
# 🌟 对齐 ViT，学习率降到 5e-4 适合微调
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=5e-4, weight_decay=0.05),
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True)
)

param_scheduler = [
    dict(type='LinearLR', begin=0, end=500, start_factor=0.001, by_epoch=False),
    dict(type='CosineAnnealingLR', T_max=50, by_epoch=True)
]

train_cfg = dict(
    by_epoch=True,
    max_epochs=50,  # 🌟 统一为 50 轮
    val_interval=5
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
        _scope_='mmdet',
        type='CSPNeXt',
        arch='P5',
        expand_ratio=0.5,
        deepen_factor=0.167,  # 这是 RTMPose-M 的参数，如果是 L 请改为 1.0
        widen_factor=0.375,   # 这是 RTMPose-M 的参数，如果是 L 请改为 1.0
        out_indices=(4, ),
        channel_attention=True,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU'),
        # 因为用了全局 load_from，这里的 init_cfg 其实可以不写，但为了规范保留
        init_cfg=dict(
            type='Pretrained',
            prefix='backbone.',
            checkpoint='https://download.openmmlab.com/mmpose/v1/projects/'
                       'rtmposev1/cspnext-tiny_udp-aic-coco_210e-256x192-cbed682d_20230130.pth'  # noqa
        )
    ),
    head=dict(
        type='CombinedRTMAnatomyAwareHead', # 🌟 使用你写的 RTM 版 Head
        in_channels=384, # 注意：RTMPose-M 这里是 768，如果是 L 则是 1024
        out_channels=31,
        input_size=codec['input_size'],
        in_featuremap_size=tuple([s // 32 for s in codec['input_size']]),
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
        type_loss_weight=0.1,
        tau=0.2,
        bio_loss_weight=0.03, # 🌟 对比损失权重
        with_contrastive=False,
        decoder=codec
    ),
    test_cfg=dict(flip_test=True)
)

# ==============================================================================
# 4. Data Pipeline (🌟 核心修正：与 ViT 完全一致的纯净 Pipeline)
# ==============================================================================
train_pipeline = [
    dict(type='LoadImage', imdecode_backend='pillow'),
    dict(type='GetBBoxCenterScale'),
    dict(type='CustomRandomFlip', direction='horizontal'),
    dict(type='ClampScale'),
    dict(type='RandomBBoxTransform'),
    # RTMPose 默认用 SimCC，通常不开启 UDP，所以 use_udp=False
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
# 5. Dataloaders
# ==============================================================================
train_dataloader = dict(
    batch_size=64, num_workers=4, persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=DATASET_TYPE, data_root=DATA_ROOT, ann_file=TRAIN_ANN,
        data_prefix=dict(img='train_final/images/'), pipeline=train_pipeline,
    )
)

val_dataloader = dict(
    batch_size=32, num_workers=4, persistent_workers=True, drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=DATASET_TYPE, data_root=DATA_ROOT, ann_file=VAL_ANN,
        data_prefix=dict(img='test_final/images/'), pipeline=val_pipeline, test_mode=True,
    )
)

test_dataloader = dict(
    batch_size=32, num_workers=4, persistent_workers=True, drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=DATASET_TYPE, data_root=DATA_ROOT, ann_file=TEST_ANN,
        data_prefix=dict(img='test_final/images/'), pipeline=val_pipeline, test_mode=True,
    )
)

# ==============================================================================
# 6. Evaluators
# ==============================================================================
val_evaluator = dict(type='ProstheticsMetric', ann_file=VAL_ANN, score_thr=0.3)
test_evaluator = dict(type='ProstheticsMetric', ann_file=TEST_ANN, score_thr=0.3)

# ==============================================================================
# 7. Hooks & Visualizer
# ==============================================================================
# 🌟 保留 EMAHook (RTMPose 微调必备)，删除了 PipelineSwitchHook
custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        priority=49)
]

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=5, max_keep_ckpts=-1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)

visualizer = dict(
    type='PoseLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(
            type='WandbVisBackend',
            init_kwargs=dict(
                project='prosthetics-pose-estimation',
                name='RTMPose-t-prosthetics_combined_loss', # 🌟 区分你的 ViT Run
                entity='qitianye1104'
            )
        )
    ],
    name='visualizer'
)