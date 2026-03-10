dataset_info = dict(
    dataset_name='ld_pros_v2_25kpts',
    # 正大光明地写 25 个点，不再需要“骗”检查了
    keypoint_info={i: dict(name=f'kp_{i}', id=i, color=[255, 0, 0], type='', swap='') for i in range(25)},
    skeleton_info={},
    joint_weights=[1.0] * 25,  # 权重数量对应 25
    # 精准对接 25 个点的 OKS 容忍度
    sigmas=[
        # --- 0-16: COCO 原生 17 个点 ---
        0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072,
        0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089,
        # --- 17-24: LDPose 的 8 个残肢点 ---
        # 对应：L/R-Elbow-Res-Above, L/R-Elbow-Res-Below, L/R-Knee-Res-Above, L/R-Knee-Res-Below
        0.072, 0.072, 0.062, 0.062, 0.087, 0.087, 0.089, 0.089
    ]
)