import numpy as np
import torch

from mmpose.registry import DATASETS
from mmpose.datasets import CocoDataset



@DATASETS.register_module()
class LDProsDataset(CocoDataset):
    """
    Custom Dataset for Amputee Pose Estimation (25 Keypoints).
    Based on user provided schema: "person_prosthesis_merged"
    """

    METAINFO = {
        'dataset_name': 'ld_pros_pose',
        'classes': ('person',),

        # === 1. 关键点定义 (完全对应截图) ===
        'keypoint_info': {
            # --- Part A: COCO 原生 17 点 (ID 0-16) ---
            0: dict(name='nose', id=0, color=[51, 153, 255], type='upper', swap=''),
            1: dict(name='left_eye', id=1, color=[51, 153, 255], type='upper', swap='right_eye'),
            2: dict(name='right_eye', id=2, color=[51, 153, 255], type='upper', swap='left_eye'),
            3: dict(name='left_ear', id=3, color=[51, 153, 255], type='upper', swap='right_ear'),
            4: dict(name='right_ear', id=4, color=[51, 153, 255], type='upper', swap='left_ear'),
            5: dict(name='left_shoulder', id=5, color=[0, 255, 0], type='upper', swap='right_shoulder'),
            6: dict(name='right_shoulder', id=6, color=[255, 128, 0], type='upper', swap='left_shoulder'),
            7: dict(name='left_elbow', id=7, color=[0, 255, 0], type='upper', swap='right_elbow'),
            8: dict(name='right_elbow', id=8, color=[255, 128, 0], type='upper', swap='left_elbow'),
            9: dict(name='left_wrist', id=9, color=[0, 255, 0], type='upper', swap='right_wrist'),
            10: dict(name='right_wrist', id=10, color=[255, 128, 0], type='upper', swap='left_wrist'),
            11: dict(name='left_hip', id=11, color=[0, 255, 0], type='lower', swap='right_hip'),
            12: dict(name='right_hip', id=12, color=[255, 128, 0], type='lower', swap='left_hip'),
            13: dict(name='left_knee', id=13, color=[0, 255, 0], type='lower', swap='right_knee'),
            14: dict(name='right_knee', id=14, color=[255, 128, 0], type='lower', swap='left_knee'),
            15: dict(name='left_ankle', id=15, color=[0, 255, 0], type='lower', swap='right_ankle'),
            16: dict(name='right_ankle', id=16, color=[255, 128, 0], type='lower', swap='left_ankle'),

            # --- Part B: 自定义残肢/假肢点 (ID 17-24) ---
            # 颜色设定：使用了红色系 [255, 0, 0] 以便在可视化时突出显示

            # 肘上 (连接到 Shoulder)
            17: dict(name='L-Elbow-Res-Above', id=17, color=[255, 0, 0], type='upper', swap='R-Elbow-Res-Above'),
            18: dict(name='R-Elbow-Res-Above', id=18, color=[255, 0, 0], type='upper', swap='L-Elbow-Res-Above'),

            # 肘下 (连接到 Elbow)
            19: dict(name='L-Elbow-Res-Below', id=19, color=[255, 0, 0], type='upper', swap='R-Elbow-Res-Below'),
            20: dict(name='R-Elbow-Res-Below', id=20, color=[255, 0, 0], type='upper', swap='L-Elbow-Res-Below'),

            # 膝上 (连接到 Hip)
            21: dict(name='L-Knee-Res-Above', id=21, color=[255, 0, 0], type='lower', swap='R-Knee-Res-Above'),
            22: dict(name='R-Knee-Res-Above', id=22, color=[255, 0, 0], type='lower', swap='L-Knee-Res-Above'),

            # 膝下 (连接到 Knee)
            23: dict(name='L-Knee-Res-Below', id=23, color=[255, 0, 0], type='lower', swap='R-Knee-Res-Below'),
            24: dict(name='R-Knee-Res-Below', id=24, color=[255, 0, 0], type='lower', swap='L-Knee-Res-Below'),
        },

        # === 2. 骨架连接 (根据 ID 和解剖逻辑推导) ===
        'skeleton_info': {
            # -- COCO 标准连线 --
            0: dict(link=('left_ankle', 'left_knee'), id=0, color=[0, 255, 0]),
            1: dict(link=('left_knee', 'left_hip'), id=1, color=[0, 255, 0]),
            2: dict(link=('right_ankle', 'right_knee'), id=2, color=[255, 128, 0]),
            3: dict(link=('right_knee', 'right_hip'), id=3, color=[255, 128, 0]),
            4: dict(link=('left_hip', 'right_hip'), id=4, color=[51, 153, 255]),
            5: dict(link=('left_shoulder', 'left_hip'), id=5, color=[51, 153, 255]),
            6: dict(link=('right_shoulder', 'right_hip'), id=6, color=[51, 153, 255]),
            7: dict(link=('left_shoulder', 'right_shoulder'), id=7, color=[51, 153, 255]),
            8: dict(link=('left_shoulder', 'left_elbow'), id=8, color=[0, 255, 0]),
            9: dict(link=('right_shoulder', 'right_elbow'), id=9, color=[255, 128, 0]),
            10: dict(link=('left_elbow', 'left_wrist'), id=10, color=[0, 255, 0]),
            11: dict(link=('right_elbow', 'right_wrist'), id=11, color=[255, 128, 0]),
            12: dict(link=('left_eye', 'right_eye'), id=12, color=[51, 153, 255]),
            13: dict(link=('nose', 'left_eye'), id=13, color=[51, 153, 255]),
            14: dict(link=('nose', 'right_eye'), id=14, color=[51, 153, 255]),
            15: dict(link=('left_eye', 'left_ear'), id=15, color=[51, 153, 255]),
            16: dict(link=('right_eye', 'right_ear'), id=16, color=[51, 153, 255]),

            # -- 自定义连线 (推测逻辑) --
            # "Res-Above" 通常是从肩膀/髋部延伸出来的
            17: dict(link=('left_shoulder', 'L-Elbow-Res-Above'), id=17, color=[255, 0, 0]),
            18: dict(link=('right_shoulder', 'R-Elbow-Res-Above'), id=18, color=[255, 0, 0]),
            19: dict(link=('left_hip', 'L-Knee-Res-Above'), id=19, color=[255, 0, 0]),
            20: dict(link=('right_hip', 'R-Knee-Res-Above'), id=20, color=[255, 0, 0]),

            # "Res-Below" 通常是从肘部/膝盖延伸出来的
            21: dict(link=('left_elbow', 'L-Elbow-Res-Below'), id=21, color=[255, 0, 0]),
            22: dict(link=('right_elbow', 'R-Elbow-Res-Below'), id=22, color=[255, 0, 0]),
            23: dict(link=('left_knee', 'L-Knee-Res-Below'), id=23, color=[255, 0, 0]),
            24: dict(link=('right_knee', 'R-Knee-Res-Below'), id=24, color=[255, 0, 0]),
        },

        # === 3. 翻转时对应的 ID 列表 ===
        # 这个列表非常关键，MMPose 训练时 Flip 增强就是靠这个 list 知道谁和谁互换
        # 格式：[1.0] * 25
        'joint_weights': [1.] * 25,

        # Sigma (用于 OKS 计算)，给自定义点一个默认值 0.05
        'sigmas': [
            0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072,
            0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089,
            0.072, 0.072, 0.062, 0.062, 0.087, 0.087, 0.089, 0.089
        ],
    }

    def parse_data_info(self, raw_data_info):
        """
        读取 JSON 中的 keypoint_types 并存入 data_info
        """
        data_info = super().parse_data_info(raw_data_info)

        # 获取 raw_ann_info (MMPose v1.x 标准结构)
        ann_info = raw_data_info.get('raw_ann_info', {})

        # 假设 JSON 里有一个 key 叫 "keypoint_types"
        if 'keypoint_types' in ann_info:
            types = np.array(ann_info['keypoint_types'], dtype=np.int64)
            data_info['keypoint_types'] = torch.from_numpy(types[None, :])
        else:
            # 默认填充 0 (假设 0 代表正常点，非0代表特殊点)
            data_info['keypoint_types'] = torch.zeros((1, 25), dtype=torch.long)

        data_info['instance_mapping_table'] = dict(
            bbox='bboxes',
            bbox_score='bbox_scores',
            keypoints='keypoints',
            keypoints_cam='keypoints_cam',
            keypoints_visible='keypoints_visible',
            bbox_scale='bbox_scales',
            head_size='head_size',
            keypoint_types='keypoint_types'
        )

        return data_info