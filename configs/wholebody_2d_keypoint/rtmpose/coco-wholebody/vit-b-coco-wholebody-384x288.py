_base_ = ['../../configs/_base_/default_runtime.py']

# ================= 1. 运行时设置 =================
# ViT 收敛比较慢，且我们要从 MAE 只有 backbone 的权重开始练
# 建议跑 100-210 epoch。为了 Benchmark 统一，这里设为 100 (和你的 RTMW 一致)
max_epochs = 100
base_lr = 5e-4 # ViT 学习率通常比 CNN 低

train_cfg = dict(max_epochs=max_epochs, val_interval=10)
randomness = dict(seed=21)

# ================= 2. 优化器 (ViT 标配 AdamW) =================
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=base_lr,
        betas=(0.9, 0.999),
        weight_decay=0.1), # ViT 需要较大的 weight decay 防止过拟合
    #以此实现 Layer-wise learning rate decay (ViT 常用技巧)
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1), # 骨干网络学习率是 Head 的 1/10
            'norm': dict(decay_mult=0.)
        }))

# 学习率调度
param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-4, by_epoch=False, begin=0, end=1000),
    dict(
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.01,
        begin=0,
        end=max_epochs,
        T_max=max_epochs,
        by_epoch=True,
        convert_to_iter_based=True),
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
        arch='base',
        img_size=(384, 288), # H, W (注意这里和 input_size 的顺序通常相反，H在前)
        patch_size=16,
        qkv_bias=True,
        drop_path_rate=0.3, # 防止过拟合
        # 【关键】加载你的 MAE 预训练权重
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmpose/v1/pretrained_models/mae_pretrain_vit_base_20230913.pth')
    ),
    head=dict(
        type='HeatmapHead',
        in_channels=768, # ViT-Base 的输出通道
        out_channels=133, # WholeBody 133点
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
    # 强制 Resize 到 384x288
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
    batch_size=32, # 【警告】ViT 显存占用大，如果 4090 爆显存，请改为 16 或 8
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
    num_workers=8,
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