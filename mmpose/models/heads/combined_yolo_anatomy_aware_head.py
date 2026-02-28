import torch
import torch.nn as nn

from mmpose.models.heads.hybrid_heads import YOLOXPoseHead
from mmpose.registry import MODELS


@MODELS.register_module()
class CombinedYOLOAnatomyAwareHead(YOLOXPoseHead):
    def __init__(self,
                 type_loss_weight=1.0,
                 tau=1.0,
                 bio_loss_weight=1.0,
                 with_contrastive=False,
                 **kwargs):
        # 初始化原生的 YOLOXPoseHead (包含 bbox, kpt 分支)
        super().__init__(**kwargs)

        self.tau = tau
        self.type_loss_weight = type_loss_weight
        self.bio_loss_weight = bio_loss_weight
        self.with_contrastive = with_contrastive
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')

        # 解剖学拓扑字典
        self.omega_dict = {
            23: [7, 9, 17, 25], 24: [8, 10, 18, 26],
            25: [9, 17, 23], 26: [10, 18, 24],
            27: [13, 15, 19, 21, 29], 28: [14, 16, 20, 22, 30],
            29: [15, 19, 21, 27], 30: [16, 20, 22, 28],
        }

    def _init_layers(self):
        """重写此方法，在原生 YOLOX 结构上追加 Type 预测分支"""
        super()._init_layers()

        # 为 YOLO 的每一个特征层 (通常是 P3, P4, P5) 增加一个并行的 1x1 卷积分支
        self.type_preds = nn.ModuleList()
        for in_c in self.in_channels:
            self.type_preds.append(
                nn.Conv2d(in_c, self.num_keypoints * 3, 1)
            )

    def forward(self, x):
        """前向传播：获取原生输出，并追加密集 Type 预测"""
        # 1. 获取原生的分类、边界框、关键点预测
        cls_scores, bbox_preds, kpt_preds = super().forward(x)

        # 2. 计算每个特征层的 Type 预测 [B, K*3, H, W]
        type_preds = [type_pred(feat) for type_pred, feat in zip(self.type_preds, x)]

        # 返回时带上 type_preds
        return cls_scores, bbox_preds, kpt_preds, type_preds

    def loss(self, x, batch_data_samples):
        """
        重写 loss 逻辑。使用中心点投影法 (Center-Mapping) 提取正样本计算 Type Loss。
        """
        outs = self.forward(x)
        cls_scores, bbox_preds, kpt_preds, type_preds = outs

        # 1. 调用父类计算基础的 YOLO 损失
        base_losses = super().loss_by_feat(cls_scores, bbox_preds, kpt_preds, batch_data_samples)
        losses = dict(**base_losses)

        device = cls_scores[0].device
        B = cls_scores[0].shape[0]
        K = self.num_keypoints

        # ------------------------------------------------------------------
        # 核心：使用中心点投影法提取正样本
        # ------------------------------------------------------------------
        # 我们使用分辨率最高、细节最好的 P3 特征图 (type_preds[0]) 计算 Type Loss
        # YOLO 系列的 P3 特征图默认 Stride 为 8
        stride = 8
        feat_h, feat_w = type_preds[0].shape[2:]

        pos_type_logits_list = []
        pos_gt_types_list = []
        pos_visible_list = []

        # 获取全局类别权重 (与之前 RTMPose 的逻辑保持完全一致)
        global_type_weights = None
        for d in batch_data_samples:
            if hasattr(d.gt_instances, 'global_type_weights'):
                global_type_weights = torch.as_tensor(
                    d.gt_instances.global_type_weights[0], dtype=torch.float32
                ).to(device)
                break
        if global_type_weights is None:
            raise ValueError('gt_instances must contain global_type_weights.')

        # 遍历批次中的每一张图
        for i in range(B):
            gt_instances = batch_data_samples[i].gt_instances
            if len(gt_instances.bboxes) == 0:
                continue

            bboxes = torch.as_tensor(gt_instances.bboxes, device=device)
            gt_types = torch.as_tensor(gt_instances.keypoint_types, device=device).long()
            visible = torch.as_tensor(gt_instances.keypoints_visible, device=device).float()

            # 计算真实框的中心点
            centers_x = (bboxes[:, 0] + bboxes[:, 2]) / 2.0
            centers_y = (bboxes[:, 1] + bboxes[:, 3]) / 2.0

            # 将中心点坐标映射到 P3 特征图的网格索引上，并限制在边界内防止越界
            grid_x = torch.clamp((centers_x / stride).long(), 0, feat_w - 1)
            grid_y = torch.clamp((centers_y / stride).long(), 0, feat_h - 1)

            # 抠图：从 P3 层特征 [K*3, H, W] 中精准提取出这些网格位置的 Logits
            img_type_logits = type_preds[0][i]
            # 索引提取后维度变为 [K*3, N_persons]，需转置为 [N_persons, K*3]
            inst_logits = img_type_logits[:, grid_y, grid_x].transpose(0, 1)
            inst_logits = inst_logits.view(-1, K, 3)  # -> [N_persons, K, 3]

            pos_type_logits_list.append(inst_logits)
            pos_gt_types_list.append(gt_types)
            pos_visible_list.append(visible)

        # ------------------------------------------------------------------
        # 计算 Anatomy Aware Loss
        # ------------------------------------------------------------------
        if len(pos_type_logits_list) > 0:
            # 拼接批次内所有提取到的有效行人
            pos_type_logits = torch.cat(pos_type_logits_list, dim=0)  # [N_total, K, 3]
            pos_gt_types = torch.cat(pos_gt_types_list, dim=0)  # [N_total, K]
            pos_visible = torch.cat(pos_visible_list, dim=0)  # [N_total, K]

            N_pos = pos_type_logits.shape[0]

            # CrossEntropy Loss (Type)
            # 按照你原始的逻辑：将 global_type_weights [K, 3] 扩展到所有行人
            expanded_type_weights = global_type_weights.unsqueeze(0).expand(N_pos, -1, -1)
            gathered_type_weights = expanded_type_weights.gather(
                dim=2, index=pos_gt_types.unsqueeze(2)
            ).squeeze(-1)

            # CE Mask 计算
            ce_mask = (pos_visible > 0).float() * gathered_type_weights
            raw_loss_type = self.ce_loss(pos_type_logits.view(-1, 3), pos_gt_types.view(-1))
            loss_type = (raw_loss_type * ce_mask.view(-1)).sum() / (ce_mask.sum() + 1e-6)
            losses['loss_type'] = self.type_loss_weight * loss_type

            # BioContrastive Loss
            if self.with_contrastive:
                type_probs = torch.softmax(pos_type_logits, dim=-1)
                p_bio = type_probs[:, :, 0]

                loss_bio_total = 0.0
                valid_r_count = 0.0

                for r, omega_r in self.omega_dict.items():
                    v_r_mask = ((pos_visible[:, r] > 0) & (pos_gt_types[:, r] == 0)).float()
                    exp_p_r = torch.exp(p_bio[:, r] / self.tau)  # [N_pos]

                    sum_exp_j = torch.zeros(N_pos, device=device)
                    for j in omega_r:
                        v_j_mask = (pos_visible[:, j] > 0).float()  # [N_pos]
                        sum_exp_j += v_j_mask * torch.exp(p_bio[:, j] / self.tau)

                    prob_r = exp_p_r / (exp_p_r + sum_exp_j + 1e-6)
                    loss_r = -torch.log(prob_r + 1e-6)  # [N_pos]

                    loss_bio_total += (loss_r * v_r_mask).sum()
                    valid_r_count += v_r_mask.sum()

                if valid_r_count > 0:
                    losses['loss_bio'] = self.bio_loss_weight * (loss_bio_total / valid_r_count)
                else:
                    losses['loss_bio'] = p_bio.sum() * 0.0
        else:
            # 防御机制：这一批全是背景
            losses['loss_type'] = sum([t.sum() * 0.0 for t in type_preds])
            if self.with_contrastive:
                losses['loss_bio'] = sum([t.sum() * 0.0 for t in type_preds])

        return losses

    # =========================================================================
    # Inference / Predict 补充
    # =========================================================================
    def predict(self, x, batch_data_samples, rescale=False):
        batch_img_metas = [data_sample.metainfo for data_sample in batch_data_samples]

        cls_scores, bbox_preds, kpt_preds, type_preds = self.forward(x)

        results_list = self.predict_by_feat(
            cls_scores, bbox_preds, kpt_preds, type_preds,
            batch_img_metas=batch_img_metas, rescale=rescale
        )
        return results_list

    def predict_by_feat(self, cls_scores, bbox_preds, kpt_preds, type_preds,
                        batch_img_metas, rescale=False, **kwargs):

        # 调用父类处理原始 bbox 和 kpt 的 NMS
        results_list = super().predict_by_feat(
            cls_scores, bbox_preds, kpt_preds,
            batch_img_metas=batch_img_metas, rescale=rescale, **kwargs
        )

        device = cls_scores[0].device
        K = self.num_keypoints
        feat_stride = 8  # 默认使用 P3 层进行反查

        for i, result in enumerate(results_list):
            pred_kpts = result.keypoints  # [N_persons, K, 2]

            if len(pred_kpts) == 0:
                result.keypoint_types = torch.empty((0, K)).to(device)
                result.type_scores = torch.empty((0, K, 3)).to(device)
                continue

            feat_h, feat_w = type_preds[0].shape[2:]

            # 使用关键点中心坐标去 P3 特征图反查
            grid_x = torch.clamp(torch.tensor(pred_kpts[..., 0]) / feat_stride, 0, feat_w - 1).long()
            grid_y = torch.clamp(torch.tensor(pred_kpts[..., 1]) / feat_stride, 0, feat_h - 1).long()

            p3_type_feat = type_preds[0][i].view(K, 3, feat_h, feat_w)

            N_persons = pred_kpts.shape[0]
            inst_type_logits = torch.zeros((N_persons, K, 3), device=device)

            for p in range(N_persons):
                for k in range(K):
                    x_idx = grid_x[p, k]
                    y_idx = grid_y[p, k]
                    inst_type_logits[p, k, :] = p3_type_feat[k, :, y_idx, x_idx]

            type_probs = torch.softmax(inst_type_logits, dim=-1)
            pred_types = torch.argmax(type_probs, dim=-1)

            result.keypoint_types = pred_types.cpu().numpy()
            result.type_scores = type_probs.cpu().numpy()

        return results_list