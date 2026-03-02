# mmpose/models/heads/anatomy_aware_head.py
import torch
import torch.nn as nn

from mmpose.evaluation import pose_pck_accuracy
from mmpose.registry import MODELS
from mmpose.models.heads import HeatmapHead
from mmpose.utils.tensor_utils import to_numpy


@MODELS.register_module()
class ClassBalancedAnatomyAwareHead(HeatmapHead):
    def __init__(self,
                 type_loss_weight=1.0,
                 detach_type_head=False,
                 **kwargs):
        super().__init__(**kwargs)

        self.type_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global Pool: [B, C, H, W] -> [B, C, 1, 1]
            nn.Flatten(),  # [B, C]
            nn.Linear(self.in_channels, self.out_channels * 3)  # [B, K*3]
        )

        self.detach_type_head = detach_type_head
        self.type_loss_weight = type_loss_weight
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, feats, with_type=False):
        x = feats[-1]

        heatmaps = self.deconv_layers(x)
        heatmaps = self.conv_layers(heatmaps)
        heatmaps = self.final_layer(heatmaps)  # [B, K, H, W]

        if not with_type:
            return heatmaps

        type_feat = x.detach() if self.detach_type_head else x
        type_logits = self.type_head(type_feat)
        type_logits = type_logits.view(-1, self.out_channels, 3)  # [B, K, 3]

        return heatmaps, type_logits

    def loss(self, feats, batch_data_samples, train_cfg=None, **kwargs):
        pred_heatmaps, pred_type_logits = self.forward(feats, with_type=True)
        losses = dict()

        B, K = pred_heatmaps.shape[0], pred_heatmaps.shape[1]
        device = pred_heatmaps.device

        gt_heatmaps = torch.stack([d.gt_fields.heatmaps for d in batch_data_samples]).to(device)
        target_visible = torch.cat([
            torch.as_tensor(d.gt_instances.keypoints_visible, dtype=torch.float32)
            for d in batch_data_samples
        ]).to(device)
        gt_types = torch.stack([d.gt_instances['keypoint_types'] for d in batch_data_samples]).to(device).long()

        custom_reg_weights = torch.cat([
            torch.as_tensor(d.gt_instances.custom_reg_weights, dtype=torch.float32)
            for d in batch_data_samples
        ]).to(device).squeeze(-1)

        global_type_weights = None
        for d in batch_data_samples:
            if hasattr(d.gt_instances, 'global_type_weights'):
                global_type_weights = torch.as_tensor(
                    d.gt_instances.global_type_weights[0], dtype=torch.float32
                ).to(device)
                break

        if global_type_weights is None:
            raise ValueError('gt_instances must contain global_type_weights.')

        visible_flat = target_visible.view(B, K)
        types_flat = gt_types.view(B, K)
        logits_flat = pred_type_logits.view(B * K, 3)

        # CrossEntropy Loss (Type)
        expanded_type_weights = global_type_weights.unsqueeze(0).expand(B, -1, -1)
        gathered_type_weights = expanded_type_weights.gather(
            dim=2, index=types_flat.unsqueeze(2)
        ).squeeze(-1)

        ce_mask = (visible_flat > 0).float() * gathered_type_weights
        ce_mask = ce_mask.view(-1)
        raw_loss_type = self.ce_loss(logits_flat, types_flat.view(-1))
        loss_type = (raw_loss_type * ce_mask).sum() / (ce_mask.sum() + 1e-6)
        losses['loss_type'] = self.type_loss_weight * loss_type

        # MSE Loss (Heatmap)
        reg_mask = (visible_flat > 0) & (types_flat != 2)
        new_target_weight = reg_mask.float().view(visible_flat.shape)
        loss_kpt = self.loss_module(pred_heatmaps, gt_heatmaps, new_target_weight)
        losses['loss_kpt'] = loss_kpt

        with torch.no_grad():
            pred_classes = torch.argmax(pred_type_logits, dim=-1)
            correct = (pred_classes == gt_types.squeeze(1))
            acc_type = (correct.float() * (visible_flat > 0).float()).sum() / ((visible_flat > 0).float().sum() + 1e-6)
            losses['acc_type'] = acc_type

        if train_cfg.get('compute_acc', True):
            _, avg_acc, _ = pose_pck_accuracy(
                output=to_numpy(pred_heatmaps),
                target=to_numpy(gt_heatmaps),
                mask=to_numpy(new_target_weight) > 0)

            acc_pose = torch.tensor(avg_acc, device=gt_heatmaps.device)
            losses.update(acc_pose=acc_pose)

        return losses

    def predict(self, feats, batch_data_samples, test_cfg=None):

        pred_heatmaps, pred_type_logits = self.forward(feats, with_type=True)

        preds = super().predict(feats, batch_data_samples, test_cfg)

        type_probs = torch.softmax(pred_type_logits, dim=2)  # [B, K, 3]
        pred_types = torch.argmax(type_probs, dim=2)  # [B, K]

        for i, pred in enumerate(preds):
            pred.keypoint_types = pred_types[i][None]
            pred.type_scores = type_probs[i][None]

        return preds