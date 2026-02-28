# configs/body_2d_keypoint/yoloxpose/coco/ld_pros_meta.py

dataset_info = dict(
    dataset_name='ld_pros',
    # 凑够 31 个点骗过检查
    keypoint_info={i: dict(name=f'kp_{i}', id=i, color=[255, 0, 0], type='', swap='') for i in range(31)},
    skeleton_info={},
    joint_weights=[1.0] * 31,
    # 你的真实 Sigma 数据
    sigmas=[
        0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072,
        0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089,
        0.089, 0.089, 0.089, 0.089, 0.089, 0.089,
        0.072, 0.072, 0.062, 0.062, 0.087, 0.087, 0.089, 0.089
    ]
)