import torch
import torch.nn as nn
from typing import Tuple
from torch import Tensor

from mmpose.registry import MODELS
from mmpose.models.heads.hybrid_heads import YOLOXPoseHead
from mmpose.utils.typing import ConfigType, OptSampleList
from mmpose.utils import reduce_mean


@MODELS.register_module()
class LDPoseYOLOHead(YOLOXPoseHead):
    def __init__(self,
                 ld_loss_weight=1.0,
                 propose_pairs=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.ld_loss_weight = ld_loss_weight

        # 互斥对的二分类交叉熵
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')

        # 完美匹配 LDPose (25 keypoints) 的互斥关系定义
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

    def loss(self,
             feats: Tuple[Tensor],
             batch_data_samples: OptSampleList,
             train_cfg: ConfigType = {}) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            feats (Tuple[Tensor]): The multi-stage features
            batch_data_samples (List[:obj:`PoseDataSample`]): The batch
                data samples
            train_cfg (dict): The runtime config for training process.
                Defaults to {}

        Returns:
            dict: A dictionary of losses.
        """

        # 1. collect & reform predictions
        cls_scores, objectnesses, bbox_preds, kpt_offsets, \
            kpt_vis = self.forward(feats)

        featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=cls_scores[0].dtype,
            device=cls_scores[0].device,
            with_stride=True)
        flatten_priors = torch.cat(mlvl_priors)

        # flatten cls_scores, bbox_preds and objectness
        flatten_cls_scores = self._flatten_predictions(cls_scores)
        flatten_bbox_preds = self._flatten_predictions(bbox_preds)
        flatten_objectness = self._flatten_predictions(objectnesses)
        flatten_kpt_offsets = self._flatten_predictions(kpt_offsets)
        flatten_kpt_vis = self._flatten_predictions(kpt_vis)
        flatten_bbox_decoded = self.decode_bbox(flatten_bbox_preds,
                                                flatten_priors[..., :2],
                                                flatten_priors[..., -1])
        flatten_kpt_decoded = self.decode_kpt_reg(flatten_kpt_offsets,
                                                  flatten_priors[..., :2],
                                                  flatten_priors[..., -1])

        # 2. generate targets
        targets = self._get_targets(flatten_priors,
                                    flatten_cls_scores.detach(),
                                    flatten_objectness.detach(),
                                    flatten_bbox_decoded.detach(),
                                    flatten_kpt_decoded.detach(),
                                    flatten_kpt_vis.detach(),
                                    batch_data_samples)
        pos_masks, cls_targets, obj_targets, obj_weights, \
            bbox_targets, bbox_aux_targets, kpt_targets, kpt_aux_targets, \
            vis_targets, vis_weights, pos_areas, pos_priors, group_indices, \
            num_fg_imgs = targets

        num_pos = torch.tensor(
            sum(num_fg_imgs),
            dtype=torch.float,
            device=flatten_cls_scores.device)
        num_total_samples = max(reduce_mean(num_pos), 1.0)

        # 3. calculate loss
        # 3.1 objectness loss
        losses = dict()

        obj_preds = flatten_objectness.view(-1, 1)
        losses['loss_obj'] = self.loss_obj(obj_preds, obj_targets,
                                           obj_weights) / num_total_samples

        if num_pos > 0:
            # 3.2 bbox loss
            bbox_preds = flatten_bbox_decoded.view(-1, 4)[pos_masks]
            losses['loss_bbox'] = self.loss_bbox(
                bbox_preds, bbox_targets) / num_total_samples

            # 3.3 keypoint loss
            kpt_preds = flatten_kpt_decoded.view(-1, self.num_keypoints,
                                                 2)[pos_masks]
            losses['loss_kpt'] = self.loss_oks(kpt_preds, kpt_targets,
                                               vis_targets, pos_areas)

            # 3.4 keypoint visibility loss
            kpt_vis_preds = flatten_kpt_vis.view(-1,
                                                 self.num_keypoints)[pos_masks]
            losses['loss_vis'] = self.loss_vis(kpt_vis_preds, vis_targets,
                                               vis_weights)

            # 3.5 classification loss
            cls_preds = flatten_cls_scores.view(-1,
                                                self.num_classes)[pos_masks]
            losses['overlaps'] = cls_targets
            cls_targets = cls_targets.pow(self.overlaps_power).detach()
            losses['loss_cls'] = self.loss_cls(cls_preds,
                                               cls_targets) / num_total_samples

            # 3.6 Limb-Deficient Loss (LDLoss) - 严格对齐 LDPose 论文公式

            # conf_logits 形状: [N_pos, num_keypoints]
            # reg_weight 形状: [N_pos, num_keypoints] (表示关键点是否可见)

            conf_logits = flatten_kpt_vis.view(-1, self.num_keypoints)[pos_masks]  # [N_pos, num_keypoints]
            reg_weight = (vis_targets > 0).float()  # [N_pos, num_keypoints]
            pair_indices = torch.tensor(self.propose_pairs, device=conf_logits.device)

            # 提取成对的 Logits 和可见度权重
            pair_logits = conf_logits[:, pair_indices]  # [N_pos, num_pairs, 2]
            pair_visible = reg_weight[:, pair_indices]  # [N_pos, num_pairs, 2]

            # 确定哪一个是 Target (y_i): 0 表示完整点，1 表示残肢点
            pair_target_class = torch.argmax(pair_visible.int(), dim=2)  # [N_pos, num_pairs]

            # 生成 Mask (M_i): 只有两者之和大于 0 时，才计算这组对比 Loss
            pair_mask = (pair_visible.sum(dim=2) > 0).float()  # [N_pos, num_pairs]

            # =================================================================
            # 核心公式：一点一点计算分子分母
            # =================================================================

            # 1. 数值稳定性操作 (防止 exp 溢出导致 NaN)
            # 取每一对 Logit 的最大值，分子分母同时减去它，数学等价，且绝对安全
            max_logits = torch.max(pair_logits, dim=2, keepdim=True)[0].detach()
            stable_logits = pair_logits - max_logits  # [N_pos, num_pairs, 2]

            # 2. 计算分子 (Numerator)
            # 提取真实存在的那个点对应的 stable_logit
            target_logits = stable_logits.gather(2, pair_target_class.unsqueeze(2)).squeeze(2)  # [N_pos, num_pairs]
            numerator = torch.exp(target_logits)  # exp(z_{y_i} - max)

            # 3. 计算分母 (Denominator)
            # 完整点与残肢点的 exp 之和
            denominator = torch.sum(torch.exp(stable_logits), dim=2)  # exp(z_0 - max) + exp(z_1 - max)

            # 4. 计算对数概率
            # 加上 1e-6 防止 log(0) 引起灾难
            log_prob = torch.log(numerator / (denominator + 1e-6))  # [N_pos, num_pairs]

            # 5. 乘以掩码并求和 ( \sum M_i * (-log_prob) )
            loss_per_pair = -log_prob * pair_mask
            sum_loss = torch.sum(loss_per_pair)

            # 6. 计算有效 Pairs 的平均值
            valid_pairs_count = torch.sum(pair_mask)
            loss_ld = sum_loss / (valid_pairs_count + 1e-6)

            losses['loss_ld'] = self.ld_loss_weight * loss_ld

            if self.use_aux_loss:
                if hasattr(self, 'loss_bbox_aux'):
                    # 3.6 auxiliary bbox regression loss
                    bbox_preds_raw = flatten_bbox_preds.view(-1, 4)[pos_masks]
                    losses['loss_bbox_aux'] = self.loss_bbox_aux(
                        bbox_preds_raw, bbox_aux_targets) / num_total_samples

                if hasattr(self, 'loss_kpt_aux'):
                    # 3.7 auxiliary keypoint regression loss
                    kpt_preds_raw = flatten_kpt_offsets.view(
                        -1, self.num_keypoints, 2)[pos_masks]
                    kpt_weights = vis_targets / vis_targets.size(-1)
                    losses['loss_kpt_aux'] = self.loss_kpt_aux(
                        kpt_preds_raw, kpt_aux_targets, kpt_weights)

        return losses