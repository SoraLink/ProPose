_base_ = ['../../../_base_/default_runtime.py']

# runtime

randomness = dict(seed=42, deterministic=False)
train_cfg = dict(max_epochs=150, val_interval=1)

custom_imports = dict(
    imports=[
        'mmpose.engine.optim_wrappers.layer_decay_optim_wrapper'
    ],
    allow_failed_imports=False
)


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

# learning policy
param_scheduler = [
    dict(type='LinearLR', begin=0, end=500, start_factor=0.001, by_epoch=False),
    dict(type='CosineAnnealingLR', T_max=150, by_epoch=True)
]

# hooks
default_hooks = dict(
    checkpoint=dict(
        interval=1,
        save_best='coco/AP',
        rule='greater',
        max_keep_ckpts=1
    )
)

# codec settings
codec = dict(
    type='UDPHeatmap', input_size=(192, 256), heatmap_size=(48, 64), sigma=2)

# model settings
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
        type='HeatmapHead',
        in_channels=768,
        out_channels=25,
        deconv_out_channels=(256, 256),
        deconv_kernel_sizes=(4, 4),
        # final_layer=dict(kernel_size=3, padding=1),
        loss=dict(
            type='LDCombinedLoss',
            loss_weight=1.0,
            heatmap_loss=dict(
                type='KeypointMSELoss',
                use_target_weight=True,
                loss_weight=1.0
            ),
            ld_loss=dict(
                type='LDLossCE',
                ld_pairs=[
                    (7, 17),  # 左肘  vs Above-left-elbow residual
                    (9, 19),  # 左腕  vs Below-left-elbow residual
                    (13, 21),  # 左膝  vs Above-left-knee residual
                    (15, 23),  # 左踝  vs Below-left-knee residual
                    (8, 18),  # 右肘  vs Above-right-elbow residual
                    (10, 20),  # 右腕  vs Below-right-elbow residual
                    (14, 22),  # 右膝  vs Above-right-knee residual
                    (16, 24),  # 右踝  vs Below-right-knee residual
                ],
                score_from='heatmap_max',
                presence_rule='tw_only',
                reduction='mean',
                alpha=0.0001
            )
        ),
        decoder=codec,
    ),
    test_cfg=dict(
        flip_test=True,
        flip_mode='heatmap',
        shift_heatmap=False,
    ))

# base dataset settings
data_root = '/home/sora/workspace/dataset/ldpose_final/'
dataset_type = 'CocoDataset'
data_mode = 'topdown'

# pipelines
train_pipeline = [
    dict(type='LoadImage', imdecode_backend='pillow'),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
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

metainfo=dict(from_file='configs/_base_/datasets/ldpose_meta.py')


# data loaders
train_dataloader = dict(
    batch_size=64,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='pros_annotations/labels_train_final.json',
        data_prefix=dict(img='ldpose_train/'),
        pipeline=train_pipeline,
        metainfo=metainfo,
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
        ann_file='pros_annotations/labels_val_final.json',
        data_prefix=dict(img='ldpose_val/'),
        test_mode=True,
        pipeline=val_pipeline,
        metainfo=metainfo,
    ))
test_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='pros_annotations/labels_test_final.json',
        data_prefix=dict(img='ldpose_test/'),
        test_mode=True,
        pipeline=val_pipeline,
        metainfo=metainfo,
    ))

# evaluators
val_evaluator = dict(
    type='CocoMetricLD',
    ann_file=data_root + 'pros_annotations/labels_val_final.json',
    conflict_map={
        # 左臂
        17: [7, 9, 19],  # Above-left-elbow residual ↔ {left elbow, left wrist, below-left-elbow residual}
        19: [9, 17],  # Below-left-elbow residual ↔ {left wrist, above-left-elbow residual}

        # 右臂
        18: [8, 10, 20],  # Above-right-elbow residual ↔ {right elbow, right wrist, below-right-elbow residual}
        20: [10, 18],  # Below-right-elbow residual ↔ {right wrist, above-right-elbow residual}

        # 左腿
        21: [13, 15, 23],  # Above-left-knee residual ↔ {left knee, left ankle, below-left-knee residual}
        23: [15, 21],  # Below-left-knee residual ↔ {left ankle, above-left-knee residual}

        # 右腿
        22: [14, 16, 24],  # Above-right-knee residual ↔ {right knee, right ankle, below-right-knee residual}
        24: [16, 22],  # Below-right-knee residual ↔ {right ankle, above-right-knee residual}
    },  # 根据你的定义改
    exist_threshold=0.5,
)
test_evaluator = dict(
    type='CocoMetricLD',
    ann_file=data_root + 'pros_annotations/labels_test_final.json',
    conflict_map={
        # 左臂
        17: [7, 9, 19],  # Above-left-elbow residual ↔ {left elbow, left wrist, below-left-elbow residual}
        19: [9, 17],  # Below-left-elbow residual ↔ {left wrist, above-left-elbow residual}

        # 右臂
        18: [8, 10, 20],  # Above-right-elbow residual ↔ {right elbow, right wrist, below-right-elbow residual}
        20: [10, 18],  # Below-right-elbow residual ↔ {right wrist, above-right-elbow residual}

        # 左腿
        21: [13, 15, 23],  # Above-left-knee residual ↔ {left knee, left ankle, below-left-knee residual}
        23: [15, 21],  # Below-left-knee residual ↔ {left ankle, above-left-knee residual}

        # 右腿
        22: [14, 16, 24],  # Above-right-knee residual ↔ {right knee, right ankle, below-right-knee residual}
        24: [16, 22],  # Below-right-knee residual ↔ {right ankle, above-right-knee residual}
    },  # 根据你的定义改
    exist_threshold=0.5,
)
