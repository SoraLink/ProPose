_base_ = [
    '../../../_base_/default_runtime.py',
    '../../../_base_/datasets/coco_wholebody.py'
]
# ================= 1. 运行时设置 =================
train_cfg = dict(max_epochs=210, val_interval=50)

# optimizer
custom_imports = dict(
    imports=['mmpose.engine.optim_wrappers.layer_decay_optim_wrapper'],
    allow_failed_imports=False)

optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=5e-4, betas=(0.9, 0.999), weight_decay=0.1),
    paramwise_cfg=dict(
        # 【修改点 1】ViT-Large 深度为 24 层 (Base 是 12)
        num_layers=24,
        layer_decay_rate=0.75,
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
        by_epoch=False),
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
codec = dict(
    type='MSRAHeatmap',
    input_size=(288, 384), # W, H
    heatmap_size=(72, 96),
    sigma=3)

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
        # 【修改点 2】架构改为 large
        arch='large',
        img_size=(384, 288), # H, W
        patch_size=16,
        qkv_bias=True,
        # 【修改点 3】Large 模型更容易过拟合，Drop Path 建议从 0.3 提升到 0.5
        drop_path_rate=0.5,
        with_cls_token=False,
        out_type='featmap',
        patch_cfg=dict(padding=2),
        # 【修改点 4】加载 MAE ViT-Large 预训练权重
        init_cfg=dict(
            type='Pretrained',
            # 使用 MMPretrain 官方提供的 MAE ViT-Large 权重
            checkpoint='https://download.openmmlab.com/mmpose/'
            'v1/pretrained_models/mae_pretrain_vit_large_20230913.pth')
    ),
    head=dict(
        type='HeatmapHead',
        # 【修改点 5】ViT-Large 的输出通道是 1024 (Base 是 768)
        in_channels=1024,
        out_channels=133,
        deconv_out_channels=(256, 256),
        deconv_kernel_sizes=(4, 4),
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
    batch_size=32,
    num_workers=8,
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
        pipeline=val_pipeline,
    ))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoWholeBodyMetric',
    ann_file=data_root + 'whole_body_annotation/coco_wholebody_val_v1.0.json')
test_evaluator = val_evaluator