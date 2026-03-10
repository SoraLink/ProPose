import torch
import torch.nn as nn

from mmpose.evaluation import pose_pck_accuracy, simcc_pck_accuracy
from mmpose.registry import MODELS
from mmpose.models.heads import RTMCCHead
from mmpose.utils.tensor_utils import to_numpy


@MODELS.register_module()
class ProPoseRTMHead(RTMCCHead):
    def __init__(self,
                 ld_loss_weight=1.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.ld_loss_weight = ld_loss_weight

        # 互斥对的二分类交叉熵
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')

        self.propose_pairs = [
            [7, 23],   # left_elbow vs L-Elbow-Res-Above
            [8, 24],   # right_elbow vs R-Elbow-Res-Above
            [9, 25],   # left_wrist vs L-Elbow-Res-Below
            [10, 26],  # right_wrist vs R-Elbow-Res-Below
            [13, 27],  # left_knee vs L-Knee-Res-Above
            [14, 28],  # right_knee vs R-Knee-Res-Above
            [15, 29],  # left_ankle vs L-Knee-Res-Below
            [16, 30]   # right_ankle vs R-Knee-Res-Below
        ]

    def forward(self, feats):
        # 纯粹依赖基础的 RTMCCHead 输出 pred_x 和 pred_y
        return super().forward(feats)

    def loss(self, feats, batch_data_samples, train_cfg=None, **kwargs):
        pred_x, pred_y = self.forward(feats)
        losses = dict()

        B, K = pred_x.shape[0], pred_x.shape[1]
        device = pred_x.device

        # 1. 获取 Ground Truths
        gt_x = torch.cat([d.gt_instance_labels.keypoint_x_labels for d in batch_data_samples], dim=0).to(device)
        gt_y = torch.cat([d.gt_instance_labels.keypoint_y_labels for d in batch_data_samples], dim=0).to(device)
        target_visible = torch.cat([
            torch.as_tensor(d.gt_instances.keypoints_visible, dtype=torch.float32)
            for d in batch_data_samples
        ]).to(device).view(B, K)

        # 获取 keypoint_types
        gt_types = torch.stack([d.gt_instances['keypoint_types'] for d in batch_data_samples]).to(device).long()
        types_flat = gt_types.view(B, K)

        # --- 2. 基础回归 Loss (SimCC) ---
        # 还原你的逻辑：使用 type 和 vis 一起决定 regression mask
        reg_mask = (target_visible > 0) & (types_flat != 2)
        new_target_weight = reg_mask.float()  # 已彻底移除 custom_reg_weights

        pred_simcc = (pred_x, pred_y)
        gt_simcc = (gt_x, gt_y)
        loss_kpt = self.loss_module(pred_simcc, gt_simcc, new_target_weight)
        losses['loss_kpt'] = loss_kpt

        # --- 3. Limb-Deficient Loss (LDLoss) ---
        # 从 SimCC 的预测中直接提取峰值作为 confidence logits
        conf_logits_x, _ = pred_x.max(dim=2)  # X轴峰值 [B, K]
        conf_logits_y, _ = pred_y.max(dim=2)  # Y轴峰值 [B, K]
        conf_logits = conf_logits_x + conf_logits_y  # 综合 confidence [B, K]

        # 提取互斥对 logits 和 visible
        pair_indices = torch.tensor(self.propose_pairs, device=device)

        pair_logits = conf_logits[:, pair_indices]  # [B, num_pairs, 2]
        pair_visible = reg_mask[:, pair_indices]  # [B, num_pairs, 2]

        # Ground truth class: 0 代表完整关节可见，1 代表残肢端点可见
        pair_target_class = torch.argmax(pair_visible.int(), dim=2)  # [B, num_pairs]

        # 掩码：如果两个点都不存在（比如被严重遮挡），则不计算这个对的 LDLoss
        pair_mask = (pair_visible.sum(dim=2) > 0).float()  # [B, num_pairs]

        flat_logits = pair_logits.view(-1, 2)
        flat_targets = pair_target_class.view(-1)
        flat_mask = pair_mask.view(-1)

        raw_loss_ld = self.ce_loss(flat_logits, flat_targets)
        loss_ld = (raw_loss_ld * flat_mask).sum() / (flat_mask.sum() + 1e-6)

        losses['loss_ld'] = self.ld_loss_weight * loss_ld

        # --- 4. 精度评估 ---
        _, avg_acc, _ = simcc_pck_accuracy(
            output=to_numpy(pred_simcc),
            target=to_numpy(gt_simcc),
            simcc_split_ratio=self.simcc_split_ratio,
            mask=to_numpy(new_target_weight) > 0,
        )
        losses.update(acc_pose=torch.tensor(avg_acc, device=device))

        return losses

    def predict(self, feats, batch_data_samples, test_cfg=None):
        return super().predict(feats, batch_data_samples, test_cfg)