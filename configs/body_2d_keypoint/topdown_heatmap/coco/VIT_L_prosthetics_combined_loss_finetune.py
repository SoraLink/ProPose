import os.path

# ==============================================================================
# 1. 基础配置与全局加载 (核心修改：全权重加载)
# ==============================================================================
_base_ = [
    '../../../_base_/default_runtime.py',
]

# 🌟 核心：直接加载你训练好的 0.897 完整模型（含 Backbone 和 Head）
load_from = './work_dirs/VIT_L_prosthetics_combined_loss/epoch_50.pth'

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
        'mmpose.models.heads.combined_anatomy_aware_head',
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
# 3. Optimizer & Scheduler (核心修改：极低学习率微调)
# ==============================================================================
optim_wrapper = dict(
    optimizer=dict(
        type='AdamW',
        lr=5e-5, # 🌟 降到原来的 1/10，防止微调时冲垮 AP
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

# 微调不需要太长的 Warmup，直接 Cosine 退火跑 5-10 个 epoch
param_scheduler = [
    dict(type='LinearLR', begin=0, end=100, start_factor=0.001, by_epoch=False),
    dict(type='CosineAnnealingLR', T_max=10, by_epoch=True)
]

train_cfg = dict(
    by_epoch=True,
    max_epochs=10,  # 🌟 跑 10 个 Epoch 足够看出 Contrast Loss 是否有效
    val_interval=1  # 🌟 每一轮都验一次，方便对比数据
)

# ==============================================================================
# 4. Model Configuration
# ==============================================================================
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
        arch='large',
        img_size=(256, 192),
        patch_size=16,
        qkv_bias=True,
        drop_path_rate=0.5,
        with_cls_token=False,
        out_type='featmap',
        patch_cfg=dict(padding=2),
        # 这里的 init_cfg 依然保留，但 load_from 的权重会自动覆盖它
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmpose/v1/pretrained_models/mae_pretrain_vit_large_20230913.pth'),
    ),
    head=dict(
        type='CombinedAnatomyAwareHead',
        in_channels=1024,
        out_channels=31,
        deconv_out_channels=(256, 256),
        deconv_kernel_sizes=(4, 4),
        loss=dict(
            type='KeypointMSELoss',
            use_target_weight=True
        ),
        decoder=codec,
        type_loss_weight=0.001,
        tau=0.2,
        bio_loss_weight=0.0003, # 🌟 开启对比损失
    ),
    test_cfg=dict(
        flip_mode='heatmap',
        shift_heatmap=False,
    )
)

# ==============================================================================
# 5. Data Pipeline (保持不变)
# ==============================================================================
train_pipeline = [
    dict(type='LoadImage', imdecode_backend='pillow'),
    dict(type='GetBBoxCenterScale'),
    dict(type='CustomRandomFlip', direction='horizontal'),
    dict(type='ClampScale'),
    dict(type='RandomBBoxTransform'),
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
# 6. Dataloaders & Evaluators (保持不变)
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
    batch_size=32, num_workers=4, persistent_workers=True, drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=DATASET_TYPE, data_root=DATA_ROOT, ann_file=VAL_ANN,
        data_prefix=dict(img='test_final/images/'), pipeline=val_pipeline, test_mode=True,
    )
)

test_dataloader = val_dataloader

val_evaluator = dict(
    type='ProstheticsMetric',
    ann_file=VAL_ANN,
    score_thr=0.3,
)
test_evaluator = val_evaluator

# ==============================================================================
# 7. Hooks & Visualizer
# ==============================================================================
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    # 🌟 每一轮都存，方便你回头挑最好的那个数据证明对比有效
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=-1),
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
                name='ViT-L-Refine_with_Contrast', # 名字改一下，方便在 Wandb 对比
                entity='qitianye1104'
            )
        )
    ],
    name='visualizer'
)