# mmpose/models/losses/anatomy_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmpose.registry import MODELS  # 注册机制


@MODELS.register_module()
class AnatomyContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1, use_target_weight=False, loss_weight=1.0):
        super(AnatomyContrastiveLoss, self).__init__()
        self.T = temperature
        self.loss_weight = loss_weight

        # 12 对互斥关系 (Res_Index, Joint_Index)
        # 请根据你的实际数据集 keypoint 顺序确认索引
        self.pairs = [
            (17, 7), (18, 8), (19, 9), (20, 10),
            (21, 13), (22, 14), (23, 15), (24, 16),
            (17, 9), (18, 10), (21, 15), (22, 16)
        ]

    def forward(self, pred_heatmaps, pred_type_logits, gt_type_labels):
        """
        pred_heatmaps: [B, K, H, W]
        pred_type_logits: [B, K, 3]
        gt_type_labels: [B, K] (0:Normal, 1:Pros, 2:Skip)
        """
        batch_size = pred_heatmaps.shape[0]
        num_kps = pred_heatmaps.shape[1]

        # 1. 基础 Heatmap Score (Global Max Pooling)
        # view平铺 -> max(dim=2) -> 取 values
        s_raw = pred_heatmaps.view(batch_size, num_kps, -1).max(dim=2)[0]

        # 2. 基础 Type Probabilities
        probs = F.softmax(pred_type_logits, dim=2)  # [B, K, 3]
        p_normal = probs[:, :, 0]
        p_skip = probs[:, :, 2]
        p_exist = 1.0 - p_skip

        total_loss = 0.0
        valid_pairs_count = 0

        for (idx_res, idx_joint) in self.pairs:
            # --- 3. 构造复合能量 ---
            # E_amputee: 残肢 Heatmap 高 且 残肢 Type 也是"存在"
            e_amputee = s_raw[:, idx_res] * p_exist[:, idx_res]

            # E_normal: 关节 Heatmap 高 且 关节 Type 是"真人"
            e_normal = s_raw[:, idx_joint] * p_normal[:, idx_joint]

            # --- 4. GT 判决 ---
            # 这里的 gt_type_labels 需要从 data_sample 里取出来
            gt_res_type = gt_type_labels[:, idx_res]
            gt_joint_type = gt_type_labels[:, idx_joint]
            res_is_missing = (gt_res_type == 2)
            joint_is_missing = (gt_joint_type == 2)
            joint_is_prosthetic = (gt_joint_type == 1)
            is_amputee_case = (gt_res_type != 2)

            targets = torch.where(is_amputee_case,
                                  torch.zeros(batch_size, dtype=torch.long, device=logits.device),
                                  torch.ones(batch_size, dtype=torch.long, device=logits.device))

            valid_mask = ~(res_is_missing & joint_is_missing)
            truncated_amputee_mask = (res_is_missing & joint_is_prosthetic)
            valid_mask = valid_mask & (~truncated_amputee_mask)

            if valid_mask.sum() == 0:
                continue

            logits = torch.stack([e_amputee, e_normal], dim=1) / self.T

            loss_per_sample = F.cross_entropy(logits, targets, reduction='none')

            current_pair_loss = (loss_per_sample * valid_mask.float()).sum() / (valid_mask.sum() + 1e-8)
            total_loss += current_pair_loss
            valid_pairs_count += 1

        if valid_pairs_count > 0:
            return self.loss_weight * (total_loss / valid_pairs_count)
        else:
            return torch.tensor(0.0, device=pred_heatmaps.device, requires_grad=True)