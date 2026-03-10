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
                 propose_pairs=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.ld_loss_weight = ld_loss_weight

        # 互斥对的二分类交叉熵
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')

        default_propose_pairs = [
            [7, 17],  # Left Elbow vs Above Left Elbow Residual
            [8, 18],  # Right Elbow vs Above Right Elbow Residual
            [9, 19],  # Left Wrist vs Below Left Elbow Residual
            [10, 20],  # Right Wrist vs Below Right Elbow Residual
            [13, 21],  # Left Knee vs Above Left Knee Residual
            [14, 22],  # Right Knee vs Above Right Knee Residual
            [15, 23],  # Left Ankle vs Below Left Knee Residual
            [16, 24]  # Right Ankle vs Below Right Knee Residual
        ]
        self.propose_pairs = propose_pairs if propose_pairs is not None else default_propose_pairs

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



        # --- 2. 基础回归 Loss (SimCC) ---
        # 还原你的逻辑：使用 type 和 vis 一起决定 regression mask
        reg_mask = (target_visible > 0)
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
        # 注意这里把 reg_mask 转成 float，方便后面计算
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

        # 计算对数概率 (加上 1e-6 防止 log(0))
        log_prob = torch.log(numerator / (denominator + 1e-6))  # [B, num_pairs]

        # 乘以掩码并求和
        loss_per_pair = -log_prob * pair_mask
        sum_loss = torch.sum(loss_per_pair)

        # 计算有效 Pairs 的平均值
        valid_pairs_count = torch.sum(pair_mask)
        loss_ld = sum_loss / (valid_pairs_count + 1e-6)

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