import os.path

_base_ = '../../../_base_/default_runtime.py'

# ==============================================================================
# 1. 基础路径与自定义模块导入
# ==============================================================================
DATASET_TYPE = 'LDProsYoloDataset'
DATA_ROOT = '/home/sora/workspace/dataset/pros_final'
DATA_MODE = 'bottomup'  # 🌟 YOLO 是 Bottom-up，这里必须改！

TRAIN_ANN = os.path.join(DATA_ROOT, 'train_final/train_final.json')
VAL_ANN = os.path.join(DATA_ROOT, 'test_final/test_final.json')
TEST_ANN = os.path.join(DATA_ROOT, 'test_final/test_final.json')
randomness = dict(seed=42, deterministic=False)

custom_imports = dict(
    imports=[
        'mmpose.evaluation.metrics.prosthetics_bottomup_metrics_baseline',  # 你的 Metric 路径 (请确保文件名一致)
        'mmpose.models.heads.combined_yolo_anatomy_aware_head',  # 🌟 我们刚刚写的 YOLOX Head 路径
        'mmpose.datasets.datasets.custom.ld_pros_yolo_dataset',  # 你的数据集路径
    ],
    allow_failed_imports=False
)

# ==============================================================================
# 2. 训练策略 (保持 YOLOX 的 300 轮策略，比 ViT 的 50 轮长，这是 YOLO 必须的)
# ==============================================================================
train_cfg = dict(
    _delete_=True,
    type='EpochBasedTrainLoop',
    max_epochs=100,
    val_interval=10,
    dynamic_intervals=[(80, 1)])  # 最后20轮(Stage2)每个epoch都验证

auto_scale_lr = dict(base_batch_size=256)

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.004, weight_decay=0.05),
    paramwise_cfg=dict(norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True),
    clip_grad=dict(max_norm=0.1, norm_type=2))

param_scheduler = [
    dict(type='QuadraticWarmupLR', by_epoch=True, begin=0, end=5, convert_to_iter_based=True),
    dict(type='CosineAnnealingLR', eta_min=0.0002, begin=5,
         T_max=80, end=80,      # 🌟 余弦退火到第 80 轮结束
         by_epoch=True, convert_to_iter_based=True),
    dict(type='ConstantLR', by_epoch=True, factor=1,
         begin=80, end=100),    # 🌟 最后 20 轮保持恒定低学习率
]

# ==============================================================================
# 3. 数据流 Pipeline (YOLOX 专属两阶段)
# ==============================================================================
input_size = (640, 640)
codec = dict(type='YOLOXPoseAnnotationProcessor', input_size=input_size)

train_pipeline_stage1 = [
    dict(type='LoadImage', imdecode_backend='pillow'),
    dict(type='Mosaic', img_scale=input_size, pad_val=114.0, pre_transform=[dict(type='LoadImage', imdecode_backend='pillow')]),
    dict(type='BottomupRandomAffine', input_size=input_size, shift_factor=0.1, rotate_factor=10,
         scale_factor=(0.75, 1.0), pad_val=114, distribution='uniform', transform_mode='perspective',
         bbox_keep_corner=False, clip_border=True),
    dict(type='YOLOXMixUp', img_scale=input_size, ratio_range=(0.8, 1.6), pad_val=114.0,
         pre_transform=[dict(type='LoadImage', imdecode_backend='pillow')]),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip'),
    dict(type='FilterAnnotations', by_kpt=True, by_box=True, keep_empty=False),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs'),
]

train_pipeline_stage2 = [
    dict(type='LoadImage', imdecode_backend='pillow'),
    dict(type='BottomupRandomAffine', input_size=input_size, shift_prob=0, rotate_prob=0, scale_prob=0,
         scale_type='long', pad_val=(114, 114, 114), bbox_keep_corner=False, clip_border=True),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip'),
    dict(type='FilterAnnotations', by_kpt=True, by_box=True, keep_empty=False),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs'),
]

val_pipeline = [
    dict(type='LoadImage', imdecode_backend='pillow'),
    dict(type='BottomupResize', input_size=input_size, pad_val=(114, 114, 114)),
    dict(
        type='PackPoseInputs',
        meta_keys=('id', 'img_id', 'img_path', 'ori_shape', 'img_shape',
                   'input_size', 'input_center', 'input_scale'))
]

# ==============================================================================
# 4. DataLoaders (使用你的自定义路径和类)
# ==============================================================================
train_dataloader = dict(
    batch_size=32,  # 注意：如果你显存爆了，改小点，同时 lr 会被 auto_scale_lr 自动按比例缩小
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
        pipeline=train_pipeline_stage1,
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

# ==============================================================================
# 5. 模型配置 (植入 Anatomy Aware Head)
# ==============================================================================
widen_factor = 0.5
deepen_factor = 0.33
num_keypoints = 31  # 🌟 必须是你数据集的点数

model = dict(
    type='BottomupPoseEstimator',
    init_cfg=dict(type='Kaiming', layer='Conv2d', a=2.23606797749979, distribution='uniform', mode='fan_in',
                  nonlinearity='leaky_relu'),
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        pad_size_divisor=32,
        mean=[0, 0, 0], std=[1, 1, 1],  # YOLOX 输入不归一化到 0-1
        batch_augments=[dict(type='BatchSyncRandomResize', random_size_range=(480, 800), size_divisor=32, interval=1)]
    ),
    backbone=dict(
        type='CSPDarknet', deepen_factor=deepen_factor, widen_factor=widen_factor, out_indices=(2, 3, 4),
        spp_kernal_sizes=(5, 9, 13),
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001), act_cfg=dict(type='Swish'),
        init_cfg=dict(type='Pretrained',
                      checkpoint='https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_s_8x8_300e_coco/yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth',
                      prefix='backbone.')
    ),
    neck=dict(
        type='YOLOXPAFPN', in_channels=[128, 256, 512], out_channels=128, num_csp_blocks=1, use_depthwise=False,
        upsample_cfg=dict(scale_factor=2, mode='nearest'), norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='Swish')
    ),
    head=dict(
        type='CombinedYOLOAnatomyAwareHead',  # 🌟 你的新 Head
        num_keypoints=num_keypoints,
        featmap_strides=(8, 16, 32),

        # 🌟 继承你在 ViT 中调好的权重比例
        type_loss_weight=1,
        tau=0.2,
        bio_loss_weight=0.0003,
        with_contrastive=False,

        head_module_cfg=dict(
            num_classes=1, in_channels=256, feat_channels=256, widen_factor=widen_factor,
            stacked_convs=2, norm_cfg=dict(type='BN', momentum=0.03, eps=0.001), act_cfg=dict(type='Swish')),
        prior_generator=dict(type='MlvlPointGenerator', offset=0, strides=[8, 16, 32]),
        assigner=dict(
            type='SimOTAAssigner',
            dynamic_k_indicator='oks',
            oks_calculator=dict(
                type='PoseOKS',
                metainfo='configs/body_2d_keypoint/yoloxpose/coco/ld_pros_meta.py'
            )
        ),
        overlaps_power=0.5,
        loss_cls=dict(type='BCELoss', reduction='sum', loss_weight=1.0),
        loss_bbox=dict(type='IoULoss', mode='square', eps=1e-16, reduction='sum', loss_weight=5.0),
        loss_obj=dict(type='BCELoss', use_target_weight=True, reduction='sum', loss_weight=1.0),
        loss_oks=dict(
            type='OKSLoss',
            reduction='none',
            metainfo='configs/body_2d_keypoint/yoloxpose/coco/ld_pros_meta.py',
            norm_target_weight=True,
            loss_weight=30.0),
        loss_vis=dict(type='BCELoss', use_target_weight=True, reduction='mean', loss_weight=1.0),
        loss_bbox_aux=dict(type='L1Loss', reduction='sum', loss_weight=1.0),
    ),
    test_cfg=dict(score_thr=0.01, nms_thr=0.65)
)

# ==============================================================================
# 6. Evaluators (注入带匈牙利匹配的 Metric)
# ==============================================================================
val_evaluator = dict(
    type='ProstheticsBottomUpMetric',
    ann_file=VAL_ANN,
    score_mode='bbox',
    nms_mode='none',
    score_thr=0.3,
    iou_thr=0.3,  # 🌟 开启匈牙利匹配过滤
)
test_evaluator = val_evaluator

# ==============================================================================
# 7. Hooks & WandB 日志记录
# ==============================================================================
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=5, max_keep_ckpts=-1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)

custom_hooks = [
    dict(type='YOLOXPoseModeSwitchHook', num_last_epochs=20, new_train_pipeline=train_pipeline_stage2, priority=48),
    dict(type='SyncNormHook', priority=48),
    dict(type='EMAHook', ema_type='ExpMomentumEMA', momentum=0.0002, update_buffers=True, strict_load=False,
         priority=49),
]

visualizer = dict(
    type='PoseLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(
            type='WandbVisBackend',  # 🌟 开启 W&B 魔法
            init_kwargs=dict(
                project='prosthetics-pose-estimation',
                name='YOLOX-S-prosthetics_combined_loss',  # 🌟 名字改成了 YOLOX-S
                entity='qitianye1104'
            )
        )
    ],
    name='visualizer'
)