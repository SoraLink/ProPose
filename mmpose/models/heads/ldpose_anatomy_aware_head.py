import torch
import torch.nn as nn

from mmpose.evaluation import pose_pck_accuracy
from mmpose.registry import MODELS
from mmpose.models.heads import HeatmapHead
from mmpose.utils.tensor_utils import to_numpy


@MODELS.register_module()
class ProPoseHeatmapHead(HeatmapHead):
    def __init__(self,
                 ld_loss_weight=1.0,
                 propose_pairs=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.ld_loss_weight = ld_loss_weight

        # 互斥对的二分类交叉熵
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')

        # 完美匹配 ld_pros_pose (31 keypoints) 的定义
        default_propose_pairs = [
            [7, 17],   # Left Elbow vs Above Left Elbow Residual
            [8, 18],   # Right Elbow vs Above Right Elbow Residual
            [9, 19],   # Left Wrist vs Below Left Elbow Residual
            [10, 20],  # Right Wrist vs Below Right Elbow Residual
            [13, 21],  # Left Knee vs Above Left Knee Residual
            [14, 22],  # Right Knee vs Above Right Knee Residual
            [15, 23],  # Left Ankle vs Below Left Knee Residual
            [16, 24]   # Right Ankle vs Below Right Knee Residual
        ]
        self.propose_pairs = propose_pairs if propose_pairs is not None else default_propose_pairs

    def forward(self, feats):
        # 纯粹依赖原生的 HeatmapHead 前向传播，返回 [B, K, H, W] 的 heatmaps
        return super().forward(feats)

    def loss(self, feats, batch_data_samples, train_cfg=None, **kwargs):
        pred_heatmaps = self.forward(feats)
        losses = dict()

        B, K, H, W = pred_heatmaps.shape
        device = pred_heatmaps.device

        # 1. 获取 Ground Truths
        gt_heatmaps = torch.stack([d.gt_fields.heatmaps for d in batch_data_samples]).to(device)
        target_visible = torch.cat([
            torch.as_tensor(d.gt_instances.keypoints_visible, dtype=torch.float32)
            for d in batch_data_samples
        ]).to(device).view(B, K)


        # --- 2. 基础回归 Loss (Heatmap MSE) ---
        # 使用 type 和 vis 一起决定 regression mask
        reg_mask = (target_visible > 0)
        new_target_weight = reg_mask.float()  # [B, K]

        loss_kpt = self.loss_module(pred_heatmaps, gt_heatmaps, new_target_weight)
        losses['loss_kpt'] = loss_kpt

        # --- 3. Limb-Deficient Loss (LDLoss) ---
        # --- 3. Limb-Deficient Loss (LDLoss) ---
        # 对于 Heatmap 架构，点的 confidence logit 是它在特征图上的最大响应值 (Peak Value)
        # 将 [B, K, H, W] 展平成 [B, K, H*W] 并取最大值，得到 [B, K]
        conf_logits = pred_heatmaps.view(B, K, -1).max(dim=2)[0]

        # 提取互斥对 logits 和 visible
        pair_indices = torch.tensor(self.propose_pairs, device=device)

        pair_logits = conf_logits[:, pair_indices]  # [B, num_pairs, 2]
        # 注意：这里把 reg_mask 转成 float，方便后面计算
        pair_visible = reg_mask[:, pair_indices].float()  # [B, num_pairs, 2]

        # Ground truth class: 0 代表完整关节可见，1 代表残肢端点可见
        pair_target_class = torch.argmax(pair_visible.int(), dim=2)  # [B, num_pairs]

        # 掩码：如果两个点都不存在（比如被严重遮挡），则不计算这个对的 LDLoss
        pair_mask = (pair_visible.sum(dim=2) > 0).float()  # [B, num_pairs]

        # =================================================================
        # 核心公式：严格对齐 LDPose 论文的张量推导版
        # =================================================================

        # 数值稳定性操作 (防止 exp 溢出导致 NaN)
        max_logits = torch.max(pair_logits, dim=2, keepdim=True)[0].detach()
        stable_logits = pair_logits - max_logits  # [B, num_pairs, 2]

        # 计算分子 (Numerator): exp(z_{y_i} - max)
        target_logits = stable_logits.gather(2, pair_target_class.unsqueeze(2)).squeeze(2)
        numerator = torch.exp(target_logits)

        # 计算分母 (Denominator): exp(z_0 - max) + exp(z_1 - max)
        denominator = torch.sum(torch.exp(stable_logits), dim=2)

        # 计算对数概率 (加上 1e-6 防止 log(0) 引起灾难)
        log_prob = torch.log(numerator / (denominator + 1e-6))  # [B, num_pairs]

        # 乘以掩码并求和 ( \sum M_i * (-log_prob) )
        loss_per_pair = -log_prob * pair_mask
        sum_loss = torch.sum(loss_per_pair)

        # 计算有效 Pairs 的平均值
        valid_pairs_count = torch.sum(pair_mask)
        loss_ld = sum_loss / (valid_pairs_count + 1e-6)

        losses['loss_ld'] = self.ld_loss_weight * loss_ld

        # --- 4. 精度评估 ---
        if train_cfg is None or train_cfg.get('compute_acc', True):
            _, avg_acc, _ = pose_pck_accuracy(
                output=to_numpy(pred_heatmaps),
                target=to_numpy(gt_heatmaps),
                mask=to_numpy(new_target_weight) > 0)

            losses.update(acc_pose=torch.tensor(avg_acc, device=device))

        return losses

    def predict(self, feats, batch_data_samples, test_cfg=None):
        # 纯净的 Heatmap 预测过程
        return super().predict(feats, batch_data_samples, test_cfg)