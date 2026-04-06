import numpy as np

from mmpose.datasets import BaseCocoStyleDataset
from mmpose.registry import DATASETS


@DATASETS.register_module()
class AmputationCOCO(BaseCocoStyleDataset):
    METAINFO = {
        'dataset_name': 'amputation_coco',
        'num_keypoints': 25,
        'keypoint_info': {
            0: dict(name='nose', id=0, color=[51, 153, 255], type='upper', swap=''),
            1: dict(name='left_eye', id=1, color=[51, 153, 255], type='upper', swap='right_eye'),
            2: dict(name='right_eye', id=2, color=[51, 153, 255], type='upper', swap='left_eye'),
            3: dict(name='left_ear', id=3, color=[51, 153, 255], type='upper', swap='right_ear'),
            4: dict(name='right_ear', id=4, color=[51, 153, 255], type='upper', swap='left_ear'),
            5: dict(name='left_shoulder', id=5, color=[0, 255, 0], type='upper', swap='right_shoulder'),
            6: dict(name='right_shoulder', id=6, color=[0, 255, 0], type='upper', swap='left_shoulder'),
            7: dict(name='left_elbow', id=7, color=[0, 255, 0], type='upper', swap='right_elbow'),
            8: dict(name='right_elbow', id=8, color=[0, 255, 0], type='upper', swap='left_elbow'),
            9: dict(name='left_wrist', id=9, color=[0, 255, 0], type='upper', swap='right_wrist'),
            10: dict(name='right_wrist', id=10, color=[0, 255, 0], type='upper', swap='left_wrist'),
            11: dict(name='left_hip', id=11, color=[255, 128, 0], type='lower', swap='right_hip'),
            12: dict(name='right_hip', id=12, color=[255, 128, 0], type='lower', swap='left_hip'),
            13: dict(name='left_knee', id=13, color=[255, 128, 0], type='lower', swap='right_knee'),
            14: dict(name='right_knee', id=14, color=[255, 128, 0], type='lower', swap='left_knee'),
            15: dict(name='left_ankle', id=15, color=[255, 128, 0], type='lower', swap='right_ankle'),
            16: dict(name='right_ankle', id=16, color=[255, 128, 0], type='lower', swap='left_ankle'),

            17: dict(name='L_Elbow_R', id=17, color=[255, 0, 0], type='upper', swap='R_Elbow_R'),
            18: dict(name='R_Elbow_R', id=18, color=[255, 0, 0], type='upper', swap='L_Elbow_R'),
            19: dict(name='L_Wrist_R', id=19, color=[255, 0, 0], type='upper', swap='R_Wrist_R'),
            20: dict(name='R_Wrist_R', id=20, color=[255, 0, 0], type='upper', swap='L_Wrist_R'),
            21: dict(name='L_Knee_R', id=21, color=[255, 0, 0], type='lower', swap='R_Knee_R'),
            22: dict(name='R_Knee_R', id=22, color=[255, 0, 0], type='lower', swap='L_Knee_R'),
            23: dict(name='L_Ankle_R', id=23, color=[255, 0, 0], type='lower', swap='R_Ankle_R'),
            24: dict(name='R_Ankle_R', id=24, color=[255, 0, 0], type='lower', swap='L_Ankle_R'),
        },
        'skeleton_info': {
            0: dict(link=('left_shoulder', 'left_elbow'), id=0, color=[0, 255, 0]),
            1: dict(link=('left_elbow', 'left_wrist'), id=1, color=[0, 255, 0]),
            2: dict(link=('right_shoulder', 'right_elbow'), id=2, color=[0, 255, 0]),
            3: dict(link=('right_elbow', 'right_wrist'), id=3, color=[0, 255, 0]),
            4: dict(link=('left_shoulder', 'right_shoulder'), id=4, color=[0, 255, 0]),
            5: dict(link=('left_shoulder', 'left_hip'), id=5, color=[0, 255, 0]),
            6: dict(link=('right_shoulder', 'right_hip'), id=6, color=[0, 255, 0]),
            7: dict(link=('left_hip', 'right_hip'), id=7, color=[0, 255, 0]),
            8: dict(link=('left_hip', 'left_knee'), id=8, color=[255, 128, 0]),
            9: dict(link=('left_knee', 'left_ankle'), id=9, color=[255, 128, 0]),
            10: dict(link=('right_hip', 'right_knee'), id=10, color=[255, 128, 0]),
            11: dict(link=('right_knee', 'right_ankle'), id=11, color=[255, 128, 0]),

            12: dict(link=('left_shoulder', 'L_Elbow_R'), id=12, color=[255, 0, 0]),
            13: dict(link=('right_shoulder', 'R_Elbow_R'), id=13, color=[255, 0, 0]),
            14: dict(link=('left_elbow', 'L_Wrist_R'), id=14, color=[255, 0, 0]),
            15: dict(link=('right_elbow', 'R_Wrist_R'), id=15, color=[255, 0, 0]),
            16: dict(link=('left_hip', 'L_Knee_R'), id=16, color=[255, 0, 0]),
            17: dict(link=('right_hip', 'R_Knee_R'), id=17, color=[255, 0, 0]),
            18: dict(link=('left_knee', 'L_Ankle_R'), id=18, color=[255, 0, 0]),
            19: dict(link=('right_knee', 'R_Ankle_R'), id=19, color=[255, 0, 0]),
        },
        'joint_weights': [1.0] * 25,
        'sigmas': np.array([
            0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072, 0.062,
            0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089,
            0.072, 0.072, 0.062, 0.062, 0.087, 0.087, 0.089, 0.089
        ])
    }

    def parse_data_info(self, raw_data_info: dict) -> dict:
        data_info = super().parse_data_info(raw_data_info)

        old_kpts = data_info['keypoints']  # [1, 17, 2]
        old_vis = data_info['keypoints_visible']  # [1, 17]

        num_instances = old_kpts.shape[0]
        new_kpts = np.zeros((num_instances, 25, 2), dtype=np.float32)
        new_vis = np.zeros((num_instances, 25), dtype=np.float32)
        new_kpts[:, :17, :] = old_kpts
        new_vis[:, :17] = old_vis

        data_info['keypoints'] = new_kpts
        data_info['keypoints_visible'] = new_vis
        data_info['ann_id'] = raw_data_info['raw_ann_info']['id']
        return data_info