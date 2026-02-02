_base_ = [
    '../../../_base_/default_runtime.py',
    '../../../_base_/datasets/coco_wholebody.py'
]
# ================= 1. 运行时设置 =================
# ViT 收敛比较慢，且我们要从 MAE 只有 backbone 的权重开始练
# 建议跑 100-210 epoch。为了 Benchmark 统一，这里设为 100 (和你的 RTMW 一致)

# runtime
train_cfg = dict(max_epochs=210, val_interval=10)

# optimizer
custom_imports = dict(
    imports=['mmpose.engine.optim_wrappers.layer_decay_optim_wrapper'],
    allow_failed_imports=False)

optim_wrapper = dict(
    optimizer=dict(
        type='AdamW', lr=5e-4, betas=(0.9, 0.999), weight_decay=0.1),
    paramwise_cfg=dict(
        num_layers=12,
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

# 自动缩放 LR (根据 batch size)
auto_scale_lr = dict(base_batch_size=512)

# ================= 3. 编解码设置 (Codec) =================
# ViT 常用 Heatmap 方法
codec = dict(
    type='MSRAHeatmap',
    input_size=(288, 384), # W, H
    heatmap_size=(72, 96), # 也就是输入尺寸的 1/4
    sigma=3) # 分辨率高，Sigma 设为 3

# ================= 4. 模型定义 =================
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type='mmpretrain.VisionTransformer',
        arch={
            'embed_dims': 384,
            'num_layers': 12,
            'num_heads': 12,
            'feedforward_channels': 384 * 4
        },
        img_size=(384, 288), # H, W (注意这里和 input_size 的顺序通常相反，H在前)
        patch_size=16,
        qkv_bias=True,
        drop_path_rate=0.1, # 防止过拟合
        with_cls_token=False,
        out_type='featmap',
        patch_cfg=dict(padding=2),
        # 【关键】加载你的 MAE 预训练权重
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmpose/v1/pretrained_models/mae_pretrain_vit_small_20230913.pth')
    ),
    neck=dict(type='FeatureMapProcessor', scale_factor=4.0, apply_relu=True),
    head=dict(
        type='HeatmapHead',
        in_channels=384, # ViT-Base 的输出通道
        out_channels=133, # WholeBody 133点
        deconv_out_channels=[],
        deconv_kernel_sizes=[],
        loss=dict(type='KeypointMSELoss', use_target_weight=True),
        decoder=codec),
    test_cfg=dict(
        flip_test=True,
        flip_mode='heatmap',
        shift_heatmap=True))

# ================= 5. 数据集设置 =================
dataset_type = 'CocoWholeBodyDataset'
data_mode = 'topdown'
data_root = './dataset/coco/'

backend_args = dict(backend='local')

# 数据处理流程
train_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(type='RandomBBoxTransform'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='Albumentation', transforms=[
        dict(type='Blur', p=0.1),
        dict(type='MedianBlur', p=0.1),
        dict(type='CoarseDropout', max_holes=1, max_height=0.4, max_width=0.4, min_holes=1, min_height=0.2, min_width=0.2, p=0.5),
    ]),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]

val_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PackPoseInputs')
]

# Dataloader
train_dataloader = dict(
    batch_size=64, # 【警告】ViT 显存占用大，如果 4090 爆显存，请改为 16 或 8
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='whole_body_annotation/coco_wholebody_train_v1.0.json',
        data_prefix=dict(img='train2017/'),
        pipeline=train_pipeline,
    ))

val_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='whole_body_annotation/coco_wholebody_val_v1.0.json',
        data_prefix=dict(img='val2017/'),
        test_mode=True,
        # bbox_file='...', # 已注释，使用 GT 框跑分
        pipeline=val_pipeline,
    ))
test_dataloader = val_dataloader

# 评估器
val_evaluator = dict(
    type='CocoWholeBodyMetric',
    ann_file=data_root + 'whole_body_annotation/coco_wholebody_val_v1.0.json')
test_evaluator = val_evaluator