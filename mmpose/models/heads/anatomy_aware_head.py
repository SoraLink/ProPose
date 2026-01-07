# mmpose/models/heads/anatomy_aware_head.py
import torch
import torch.nn as nn

from mmpose.evaluation import pose_pck_accuracy
from mmpose.registry import MODELS
from mmpose.models.heads import HeatmapHead  # 继承现有的 Head
from mmpose.models.losses import KeypointMSELoss
from mmpose.utils.tensor_utils import to_numpy


@MODELS.register_module()
class AnatomyAwareHead(HeatmapHead):
    def __init__(self,
                 contrast_loss=None,
                 type_loss_weight=1.0,
                 **kwargs):
        super().__init__(**kwargs)

        # 1. 增加一个分支用于预测 Type (Batch, K, 3)
        # 假设输入 feature map 是 C 通道
        # 这里的 in_channels 需要和 backbone 输出对齐 (ViT-Large 可能是 1024 或 768)
        self.type_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global Pool: [B, C, H, W] -> [B, C, 1, 1]
            nn.Flatten(),  # [B, C]
            nn.Linear(self.in_channels, self.out_channels * 3)  # [B, K*3]
        )

        # 2. 初始化自定义 Contrast Loss
        self.contrast_loss_module = None
        if contrast_loss:
            self.contrast_loss_module = MODELS.build(contrast_loss)

        self.type_loss_weight = type_loss_weight
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, feats, with_type=False):
        """同时输出 Heatmap 和 Type Logits"""
        # feats 是一个 tuple，通常取最后一个 feature map
        x = feats[-1]

        # A. 原有分支：预测 Heatmaps
        heatmaps = self.deconv_layers(x)
        heatmaps = self.conv_layers(heatmaps)
        heatmaps = self.final_layer(heatmaps)  # [B, K, H, W]

        if not with_type:
            return heatmaps

        # B. 新分支：预测 Type
        # 注意：这里简单的用了一个 Linear 层。
        # 如果效果不好，可以用更复杂的卷积头。
        type_logits = self.type_head(x)
        type_logits = type_logits.view(-1, self.out_channels, 3)  # [B, K, 3]

        return heatmaps, type_logits

    def loss(self, feats, batch_data_samples, train_cfg=None, **kwargs):
        """计算混合 Loss"""
        # 1. 前向传播
        pred_heatmaps, pred_type_logits = self.forward(feats, with_type=True)

        losses = dict()

        # 2. 解析 GT 数据
        # MMPose v1.x 把 GT 放在 batch_data_samples 里
        gt_heatmaps = torch.stack([d.gt_fields.heatmaps for d in batch_data_samples]).to(pred_heatmaps.device)
        target_weight = torch.cat([d.gt_instance_labels.keypoint_weights for d in batch_data_samples]).to(
            pred_heatmaps.device)

        # === 关键：获取 GT Type ===
        # 你需要在 Dataset 里把 'type' 读进去，放在 gt_instance_labels 里
        # 假设形状是 [B, K]
        gt_types = torch.stack([d.gt_instances['keypoint_types'] for d in batch_data_samples]).to(
            pred_heatmaps.device).long()

        # 3. 计算原来的 MSE Loss (Heatmap)
        # 使用父类的 loss module (KeypointMSELoss)
        loss_kpt = self.loss_module(pred_heatmaps, gt_heatmaps, target_weight)
        losses['loss_kpt'] = loss_kpt

        # 4. 计算 Type Classification Loss (Cross Entropy)
        # 展平计算 [B*K, 3] vs [B*K]
        loss_type = self.ce_loss(pred_type_logits.view(-1, 3), gt_types.view(-1))
        losses['loss_type'] = self.type_loss_weight * loss_type

        with torch.no_grad():
            pred_classes = torch.argmax(pred_type_logits, dim=-1)
            correct = (pred_classes == gt_types.squeeze(1))
            acc_type = correct.float().mean()
            losses['acc_type'] = acc_type

        # 5. 计算 Anatomy Contrastive Loss
        if self.contrast_loss_module is not None:
            loss_contrast = self.contrast_loss_module(pred_heatmaps, pred_type_logits, gt_types)
            losses['loss_contrast'] = loss_contrast

        if train_cfg.get('compute_acc', True):
            _, avg_acc, _ = pose_pck_accuracy(
                output=to_numpy(pred_heatmaps),
                target=to_numpy(gt_heatmaps),
                mask=to_numpy(target_weight) > 0)

            acc_pose = torch.tensor(avg_acc, device=gt_heatmaps.device)
            losses.update(acc_pose=acc_pose)

        return losses

    def predict(self, feats, batch_data_samples, test_cfg=None):
        """
        推断时的逻辑。
        MMPose 默认只返回 heatmap decode 后的坐标。
        我们可以重写这个，把 Type 也塞进结果里。
        """
        pred_heatmaps, pred_type_logits = self.forward(feats, with_type=True)

        # 调用父类的 predict 获取坐标
        preds = super().predict(feats, batch_data_samples, test_cfg)

        # 把 Type 预测结果附加上去
        type_probs = torch.softmax(pred_type_logits, dim=2)  # [B, K, 3]
        pred_types = torch.argmax(type_probs, dim=2)  # [B, K]

        for i, pred in enumerate(preds):
            # 将 type 保存到 instance data 中
            pred.keypoint_types = pred_types[i][None]
            pred.type_scores = type_probs[i][None]

        return preds