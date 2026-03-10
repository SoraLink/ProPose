from mmpose.registry import DATASETS
from ..base import BaseCocoStyleDataset
import numpy as np
def get_swap(i):
    # 定义你的左右对调逻辑，防止 RandomFlip 报错
    mapping = {1: 2, 2: 1, 3: 4, 4: 3, 5: 6, 6: 5, 7: 8, 8: 7, 9: 10, 10: 9, 11: 12, 12: 11, 13: 14, 14: 13, 15: 16,
               16: 15,
               17: 18, 18: 17, 19: 20, 20: 19, 21: 22, 22: 21, 23: 24, 24: 23}
    return f'kp_{mapping[i]}' if i in mapping else ''

@DATASETS.register_module()
class LDDataset(BaseCocoStyleDataset):
    """自定义 ProPose 25点数据集类"""

    METAINFO = dict(
        dataset_name='ld_pros_25kpts_v3',
        num_keypoints=25,
        sigmas=np.array([
            # --- 0-16: COCO 17个点 ---
            0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072,
            0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089,
            # --- 17-24: 你的 8个残肢点 (L/R Elbow/Knee Above/Below) ---
            0.072, 0.072, 0.062, 0.062, 0.087, 0.087, 0.089, 0.089
        ]),
        keypoint_info={
            i: dict(name=f'kp_{i}', id=i, color=[255, 0, 0], type='', swap=get_swap(i))
            for i in range(25)
        },
        skeleton_info={},
        joint_weights=np.ones((25, 1), dtype=np.float32)
    )
