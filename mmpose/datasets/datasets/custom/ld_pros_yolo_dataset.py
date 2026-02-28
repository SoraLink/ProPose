import os

import numpy as np
import torch
from pycocotools.coco import COCO

from mmpose.registry import DATASETS
from mmpose.datasets import CocoDataset



@DATASETS.register_module()
class LDProsYoloDataset(CocoDataset):
    """
    Custom Dataset for Amputee Pose Estimation (25 Keypoints).
    Based on user provided schema: "person_prosthesis_merged"
    """

    METAINFO = {
        'dataset_name': 'ld_pros_pose',
        'classes': ('person',),
        'num_keypoints': 31,
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
            17: dict(name='L_Middle_Tip', id=17, color=[255, 0, 255], type='upper', swap='R_Middle_Tip'),
            18: dict(name='R_Middle_Tip', id=18, color=[255, 0, 255], type='upper', swap='L_Middle_Tip'),
            19: dict(name='L_Heel', id=19, color=[255, 0, 255], type='lower', swap='R_Heel'),
            20: dict(name='R_Heel', id=20, color=[255, 0, 255], type='lower', swap='L_Heel'),
            21: dict(name='L_Toe_Tip', id=21, color=[255, 0, 255], type='lower', swap='R_Toe_Tip'),
            22: dict(name='R_Toe_Tip', id=22, color=[255, 0, 255], type='lower', swap='L_Toe_Tip'),

            # 23-30: 残肢点 (Res KPs)
            23: dict(name='L-Elbow-Res-Above', id=23, color=[255, 0, 0], type='upper', swap='R-Elbow-Res-Above'),
            24: dict(name='R-Elbow-Res-Above', id=24, color=[255, 0, 0], type='upper', swap='L-Elbow-Res-Above'),
            25: dict(name='L-Elbow-Res-Below', id=25, color=[255, 0, 0], type='upper', swap='R-Elbow-Res-Below'),
            26: dict(name='R-Elbow-Res-Below', id=26, color=[255, 0, 0], type='upper', swap='L-Elbow-Res-Below'),
            27: dict(name='L-Knee-Res-Above', id=27, color=[255, 0, 0], type='lower', swap='R-Knee-Res-Above'),
            28: dict(name='R-Knee-Res-Above', id=28, color=[255, 0, 0], type='lower', swap='L-Knee-Res-Above'),
            29: dict(name='L-Knee-Res-Below', id=29, color=[255, 0, 0], type='lower', swap='R-Knee-Res-Below'),
            30: dict(name='R-Knee-Res-Below', id=30, color=[255, 0, 0], type='lower', swap='L-Knee-Res-Below'),
        },

        # === 2. 骨架连接 (根据 ID 和解剖逻辑推导) ===
        'skeleton_info': {
            # --- Part A: 基础连线 (0-16 为原生或基础结构) ---
            0: dict(link=('nose', 'left_eye'), id=0, color=[51, 153, 255]),
            1: dict(link=('nose', 'right_eye'), id=1, color=[51, 153, 255]),
            2: dict(link=('left_eye', 'left_ear'), id=2, color=[51, 153, 255]),
            3: dict(link=('right_eye', 'right_ear'), id=3, color=[51, 153, 255]),
            4: dict(link=('left_shoulder', 'right_shoulder'), id=4, color=[51, 153, 255]),
            5: dict(link=('left_shoulder', 'left_elbow'), id=5, color=[0, 255, 0]),
            6: dict(link=('left_elbow', 'left_wrist'), id=6, color=[0, 255, 0]),
            7: dict(link=('right_shoulder', 'right_elbow'), id=7, color=[255, 128, 0]),
            8: dict(link=('right_elbow', 'right_wrist'), id=8, color=[255, 128, 0]),
            9: dict(link=('left_shoulder', 'left_hip'), id=9, color=[51, 153, 255]),
            10: dict(link=('right_shoulder', 'right_hip'), id=10, color=[51, 153, 255]),
            11: dict(link=('left_hip', 'right_hip'), id=11, color=[51, 153, 255]),
            12: dict(link=('left_hip', 'left_knee'), id=12, color=[0, 255, 0]),
            13: dict(link=('left_knee', 'left_ankle'), id=13, color=[0, 255, 0]),
            14: dict(link=('right_hip', 'right_knee'), id=14, color=[255, 128, 0]),
            15: dict(link=('right_knee', 'right_ankle'), id=15, color=[255, 128, 0]),

            # --- Part B: 新增点连线 (17-22: 肢体末端) ---
            16: dict(link=('left_wrist', 'L_Middle_Tip'), id=16, color=[0, 255, 255]),
            17: dict(link=('right_wrist', 'R_Middle_Tip'), id=17, color=[255, 0, 255]),
            18: dict(link=('left_ankle', 'L_Heel'), id=18, color=[0, 255, 255]),
            19: dict(link=('left_ankle', 'L_Toe_Tip'), id=19, color=[0, 255, 255]),
            20: dict(link=('right_ankle', 'R_Heel'), id=20, color=[255, 0, 255]),
            21: dict(link=('right_ankle', 'R_Toe_Tip'), id=21, color=[255, 0, 255]),

            # --- Part C: 残肢连线 (23-30: 对应 RES_KPS) ---
            22: dict(link=('left_shoulder', 'L-Elbow-Res-Above'), id=22, color=[255, 0, 0]),
            23: dict(link=('right_shoulder', 'R-Elbow-Res-Above'), id=23, color=[255, 0, 0]),
            24: dict(link=('left_elbow', 'L-Elbow-Res-Below'), id=24, color=[255, 0, 0]),
            25: dict(link=('right_elbow', 'R-Elbow-Res-Below'), id=25, color=[255, 0, 0]),
            26: dict(link=('left_hip', 'L-Knee-Res-Above'), id=26, color=[255, 0, 0]),
            27: dict(link=('right_hip', 'R-Knee-Res-Above'), id=27, color=[255, 0, 0]),
            28: dict(link=('left_knee', 'L-Knee-Res-Below'), id=28, color=[255, 0, 0]),
            29: dict(link=('right_knee', 'R-Knee-Res-Below'), id=29, color=[255, 0, 0]),
        },

        # === 3. 翻转时对应的 ID 列表 ===
        # 这个列表非常关键，MMPose 训练时 Flip 增强就是靠这个 list 知道谁和谁互换
        # 格式：[1.0] * 25
        'joint_weights': [1.] * 31,

        # Sigma (用于 OKS 计算)，给自定义点一个默认值 0.05
        'sigmas': [
            0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072,
            0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089,
            0.089, 0.089, 0.089, 0.089, 0.089, 0.089,
            0.072, 0.072, 0.062, 0.062, 0.087, 0.087, 0.089, 0.089
        ],
    }

    def __init__(self, **kwargs):
        # 1. 绝对强制获取参数，没有任何 Fallback
        if 'data_root' not in kwargs or not kwargs['data_root']:
            raise ValueError(f"❌ Initialization Failed: '{self.__class__.__name__}' requires a valid 'data_root'.")

        if 'ann_file' not in kwargs or not kwargs['ann_file']:
            raise ValueError(f"❌ Initialization Failed: '{self.__class__.__name__}' requires a valid 'ann_file'.")

        data_root = kwargs['data_root']
        ann_file = kwargs['ann_file']
        full_ann_file = os.path.join(data_root, ann_file)

        # 2. 严格校验文件是否存在
        if not os.path.exists(full_ann_file):
            raise FileNotFoundError(f"❌ Annotation file does not exist: '{full_ann_file}'. "
                                    f"Cannot compute Class-Balanced Weights!")

        # 3. 使用标准的 COCO API 提前读取标注并算好权重
        print(f"[{self.__class__.__name__}] Loading annotations via pycocotools for weight calculation...")
        temp_coco = COCO(full_ann_file)
        self._compute_cb_weights(temp_coco)

        # 4. 权重表准备就绪，按动启动按钮，交由父类执行后续流程
        super().__init__(**kwargs)

    def _compute_cb_weights(self, coco_api, beta=0.999, reg_weight_cap=5.0):
        """
        利用 pycocotools 传入的 coco_api 统计点位比例
        """
        print(f"[{self.__class__.__name__}] Calculating Global Class-Balanced Weights...")
        num_kpts = self.METAINFO['num_keypoints']
        counts = np.zeros((num_kpts, 3), dtype=int)

        # 1. 全局统计：直接复用你最初写的那套优雅的 coco.anns.values() 遍历
        for ann in coco_api.anns.values():
            if 'keypoints' not in ann or 'keypoint_types' not in ann:
                continue

            kps = np.array(ann['keypoints']).reshape(-1, 3)
            vis = kps[:, 2]
            types = np.array(ann['keypoint_types'])

            for k in range(num_kpts):
                if vis[k] > 0:
                    t = types[k]
                    if 0 <= t <= 2:
                        counts[k, t] += 1

        # 辅助函数：计算有效样本量倒数
        def get_type_w(n):
            if n == 0: return 0.0
            return 1.0 / np.sqrt(n)

        self.W_type_table = np.zeros((num_kpts, 3), dtype=np.float32)
        self.W_reg_table = np.zeros((num_kpts, 2), dtype=np.float32)

        # 2. 计算 Semantic Type 的权重表 (31x3)
        for k in range(num_kpts):
            for c in range(3):
                if k >= 23 and c == 1:
                    print(f"[{self.__class__.__name__}] Skipping RES KP {k} (Type 2) for Weight Calculation.")
                    continue
                self.W_type_table[k, c] = get_type_w(counts[k, c])

            # 点内归一化
            valid_mask = self.W_type_table[k] > 0
            if valid_mask.sum() > 0:
                self.W_type_table[k, valid_mask] /= self.W_type_table[k, valid_mask].mean()

        # 3. 计算 Regression 的权重表 (31x2, 仅含 0 和 1)
        basic_total_counts = counts[0:23, 0] + counts[0:23, 1]
        avg_basic_count = np.mean(basic_total_counts)
        anchor_eff = (1.0 - np.power(beta, avg_basic_count)) / (1.0 - beta)

        for k in range(num_kpts):
            if k < 23:
                # 基础点 (0-22): 点内平衡

                c_norm = max(1, counts[k, 0])
                c_pros = max(1, counts[k, 1])

                w_norm = np.sqrt(avg_basic_count / c_norm)
                w_pros = np.sqrt(avg_basic_count / c_pros) if counts[k, 1] > 0 else 0.0


                self.W_reg_table[k, 0] = min(w_norm, reg_weight_cap)
                self.W_reg_table[k, 1] = min(w_pros, reg_weight_cap)
            else:
                # 残肢点 (23-30): 点间平衡
                c_norm = max(1, counts[k, 0])
                w_norm = np.sqrt(avg_basic_count / c_norm)

                self.W_reg_table[k, 0] = min(w_norm, reg_weight_cap)
                self.W_reg_table[k, 1] = 0.0

        self.global_type_weights = torch.from_numpy(self.W_type_table).float()
        print(f"[{self.__class__.__name__}] Weights Calculation Completed.")

    def parse_data_info(self, raw_data_info):
        """
        读取 JSON 中的 keypoint_types 并将其编码进 keypoints_visible 中，
        以此绕过 Mosaic 等数据增强机制对自定义字典键的清理限制。
        """
        data_info = super().parse_data_info(raw_data_info)

        # 获取 raw_ann_info (MMPose v1.x 标准结构)
        ann_info = raw_data_info.get('raw_ann_info', {})

        # 获取分类 Type
        if 'keypoint_types' in ann_info:
            types = np.array(ann_info['keypoint_types'], dtype=np.int64)
        else:
            raise ValueError('keypoint_types not found in raw_ann_info')

        # 获取当前人的关键点可视度，通常形状是 [1, 31]，取 [0] 变成 [31]
        vis = data_info['keypoints_visible'][0]

        # 🌟🌟🌟 核心魔术：穿马甲 (仅对原本可视度 > 0 的点进行编码) 🌟🌟🌟
        # 公式: vis_encoded = vis + type * 10
        vis_encoded = np.where(vis > 0, vis + types * 10, 0)

        # 将编码后的值覆盖回去
        data_info['keypoints_visible'][0] = vis_encoded

        # 🌟 保持极度干净：把之前 custom_weights 的打包代码全删了！
        # 只要标准的键，系统就不会在拼接图片时崩溃
        data_info['instance_mapping_table'] = dict(
            bbox='bboxes',
            bbox_score='bbox_scores',
            keypoints='keypoints',
            keypoints_cam='keypoints_cam',
            keypoints_visible='keypoints_visible',
            bbox_scale='bbox_scales',
            head_size='head_size',
        )

        return data_info