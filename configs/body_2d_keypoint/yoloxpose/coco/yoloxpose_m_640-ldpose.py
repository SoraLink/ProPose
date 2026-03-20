_base_ = './yoloxpose_s-640-ldpose.py'

widen_factor = 0.75
deepen_factor = 0.67
checkpoint = 'https://download.openmmlab.com/mmpose/v1/pretrained_models/' \
             'yolox_m_8x8_300e_coco_20230829.pth'

# model settings
model = dict(
    backbone=dict(
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        init_cfg=dict(checkpoint=checkpoint),
    ),
    neck=dict(in_channels=[192, 384, 768], out_channels=192, num_csp_blocks=2),
    head=dict(head_module_cfg=dict(widen_factor=widen_factor)))


visualizer = dict(
    type='PoseLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(
            type='WandbVisBackend',  # 🌟 开启 W&B 魔法
            init_kwargs=dict(
                project='prosthetics-pose-estimation',
                name='YOLOX-L-prosthetics_combined_loss-ldpose',  # 🌟 名字改成了 YOLOX-S
                entity='qitianye1104'
            )
        )
    ],
    name='visualizer'
)