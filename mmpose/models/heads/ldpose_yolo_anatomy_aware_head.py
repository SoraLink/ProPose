import copy
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.structures import InstanceData
from torch import Tensor

from mmpose.models.heads.hybrid_heads import YOLOXPoseHead
from mmpose.registry import MODELS
from mmpose.structures import PoseDataSample
from mmpose.utils import reduce_mean
from mmpose.utils.typing import (ConfigType, Features, OptSampleList,
                                 Predictions, SampleList)
from mmpose.models.utils import filter_scores_and_topk
from mmpose.evaluation.functional import nms_torch


@MODELS.register_module()
class ProPoseYOLOHead(YOLOXPoseHead):
    def __init__(self,
                 ld_loss_weight=1.0,
                 propose_pairs=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.ld_loss_weight = ld_loss_weight

        # 互斥对的二分类交叉熵
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')

        # 完美匹配 ld_pros_pose (31 keypoints) 的互斥关系定义
        default_propose_pairs = [
            [7, 23],  # left_elbow vs L-Elbow-Res-Above
            [8, 24],  # right_elbow vs R-Elbow-Res-Above
            [9, 25],  # left_wrist vs L-Elbow-Res-Below
            [10, 26],  # right_wrist vs R-Elbow-Res-Below
            [13, 27],  # left_knee vs L-Knee-Res-Above
            [14, 28],  # right_knee vs R-Knee-Res-Above
            [15, 29],  # left_ankle vs L-Knee-Res-Below
            [16, 30]  # right_ankle vs R-Knee-Res-Below
        ]

        self.propose_pairs = propose_pairs if propose_pairs is not None else default_propose_pairs

    def forward(self, feats: Features):
        # 纯粹调用原生的 forward，只拿 5 个基础输出
        cls_scores, objectnesses, bbox_preds, kpt_offsets, kpt_vis = self.head_module(feats)
        return cls_scores, objectnesses, bbox_preds, kpt_offsets, kpt_vis

    def loss(self,
             feats: Tuple[Tensor],
             batch_data_samples: OptSampleList,
             train_cfg: ConfigType = {}) -> dict:

        # 1. 提取原生预测
        cls_scores, objectnesses, bbox_preds, kpt_offsets, kpt_vis = self.forward(feats)

        featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=cls_scores[0].dtype,
            device=cls_scores[0].device,
            with_stride=True)
        flatten_priors = torch.cat(mlvl_priors)

        # flatten outputs
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

        # 2. 生成 Target
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
            num_fg_imgs, type_targets = targets

        num_pos = torch.tensor(
            sum(num_fg_imgs),
            dtype=torch.float,
            device=flatten_cls_scores.device)
        num_total_samples = max(reduce_mean(num_pos), 1.0)

        losses = dict()

        # 3.1 Objectness loss
        obj_preds = flatten_objectness.view(-1, 1)
        losses['loss_obj'] = self.loss_obj(obj_preds, obj_targets, obj_weights) / num_total_samples

        if num_pos > 0:
            # 3.2 BBox loss
            bbox_preds = flatten_bbox_decoded.view(-1, 4)[pos_masks]
            losses['loss_bbox'] = self.loss_bbox(bbox_preds, bbox_targets) / num_total_samples

            # 3.3 Keypoint Regression Loss
            kpt_preds = flatten_kpt_decoded.view(-1, self.num_keypoints, 2)[pos_masks]
            # 这里是重点：利用 vis_targets 和 type_targets (如果是 2 则忽略) 作为回归掩码
            valid_mask = (vis_targets > 0) & (type_targets != 2)
            reg_weight = valid_mask.float()

            losses['loss_kpt'] = self.loss_oks(kpt_preds, kpt_targets, reg_weight, pos_areas)

            # 3.4 Keypoint Visibility Loss
            kpt_vis_preds = flatten_kpt_vis.view(-1, self.num_keypoints)[pos_masks]
            losses['loss_vis'] = self.loss_vis(kpt_vis_preds, valid_mask, vis_weights)

            # 3.5 Classification loss
            cls_preds = flatten_cls_scores.view(-1, self.num_classes)[pos_masks]
            losses['overlaps'] = cls_targets
            cls_targets = cls_targets.pow(self.overlaps_power).detach()
            losses['loss_cls'] = self.loss_cls(cls_preds, cls_targets) / num_total_samples

            # 3.6 Limb-Deficient Loss (LDLoss)
            # 对于 YOLO，直接取其预测的关键点可见度(kpt_vis_preds) 作为 confidence logits 进行压制
            conf_logits = kpt_vis_preds  # [N_pos, num_keypoints]

            pair_indices = torch.tensor(self.propose_pairs, device=conf_logits.device)

            pair_logits = conf_logits[:, pair_indices]  # [N_pos, num_pairs, 2]
            pair_visible = valid_mask[:, pair_indices]  # [N_pos, num_pairs, 2] 保持一致性用有效点作为掩码

            pair_target_class = torch.argmax(pair_visible.int(), dim=2)  # [N_pos, num_pairs]
            pair_mask = (pair_visible.sum(dim=2) > 0).float()  # [N_pos, num_pairs]

            flat_logits = pair_logits.view(-1, 2)
            flat_targets = pair_target_class.view(-1)
            flat_mask = pair_mask.view(-1)

            raw_loss_ld = self.ce_loss(flat_logits, flat_targets)
            loss_ld = (raw_loss_ld * flat_mask).sum() / (flat_mask.sum() + 1e-6)

            losses['loss_ld'] = self.ld_loss_weight * loss_ld

            # 辅助分支
            if self.use_aux_loss:
                if hasattr(self, 'loss_bbox_aux'):
                    bbox_preds_raw = flatten_bbox_preds.view(-1, 4)[pos_masks]
                    losses['loss_bbox_aux'] = self.loss_bbox_aux(
                        bbox_preds_raw, bbox_aux_targets) / num_total_samples

                if hasattr(self, 'loss_kpt_aux'):
                    kpt_preds_raw = flatten_kpt_offsets.view(-1, self.num_keypoints, 2)[pos_masks]
                    kpt_weights = vis_targets / vis_targets.size(-1)
                    losses['loss_kpt_aux'] = self.loss_kpt_aux(
                        kpt_preds_raw, kpt_aux_targets, kpt_weights)

        return losses

    @torch.no_grad()
    def _get_targets(
            self,
            priors: Tensor,
            batch_cls_scores: Tensor,
            batch_objectness: Tensor,
            batch_decoded_bboxes: Tensor,
            batch_decoded_kpts: Tensor,
            batch_kpt_vis: Tensor,
            batch_data_samples: SampleList,
    ):
        num_imgs = len(batch_data_samples)

        batch_cls_scores = batch_cls_scores.clip(min=-1e4, max=1e4).sigmoid()
        batch_objectness = batch_objectness.clip(min=-1e4, max=1e4).sigmoid()
        batch_kpt_vis = batch_kpt_vis.clip(min=-1e4, max=1e4).sigmoid()
        batch_cls_scores[torch.isnan(batch_cls_scores)] = 0
        batch_objectness[torch.isnan(batch_objectness)] = 0

        targets_each = []
        for i in range(num_imgs):
            target = self._get_targets_single(priors, batch_cls_scores[i],
                                              batch_objectness[i],
                                              batch_decoded_bboxes[i],
                                              batch_decoded_kpts[i],
                                              batch_kpt_vis[i],
                                              batch_data_samples[i])
            targets_each.append(target)

        targets = list(zip(*targets_each))
        for i, target in enumerate(targets):
            if torch.is_tensor(target[0]):
                target = tuple(filter(lambda x: x.size(0) > 0, target))
                if len(target) > 0:
                    targets[i] = torch.cat(target)

        foreground_masks, cls_targets, obj_targets, obj_weights, \
            bbox_targets, kpt_targets, vis_targets, vis_weights, pos_areas, \
            pos_priors, group_indices, num_pos_per_img, type_targets = targets

        if self.use_aux_loss:
            bbox_cxcy = (bbox_targets[:, :2] + bbox_targets[:, 2:]) / 2.0
            bbox_wh = bbox_targets[:, 2:] - bbox_targets[:, :2]
            bbox_aux_targets = torch.cat([
                (bbox_cxcy - pos_priors[:, :2]) / pos_priors[:, 2:],
                torch.log(bbox_wh / pos_priors[:, 2:] + 1e-8)
            ], dim=-1)
            kpt_aux_targets = (kpt_targets - pos_priors[:, None, :2]) / pos_priors[:, None, 2:]
        else:
            bbox_aux_targets, kpt_aux_targets = None, None

        return (foreground_masks, cls_targets, obj_targets, obj_weights,
                bbox_targets, bbox_aux_targets, kpt_targets, kpt_aux_targets,
                vis_targets, vis_weights, pos_areas, pos_priors, group_indices,
                num_pos_per_img, type_targets)

    @torch.no_grad()
    def _get_targets_single(
            self,
            priors: Tensor,
            cls_scores: Tensor,
            objectness: Tensor,
            decoded_bboxes: Tensor,
            decoded_kpts: Tensor,
            kpt_vis: Tensor,
            data_sample: PoseDataSample,
    ) -> tuple:
        num_priors = priors.size(0)
        gt_instances = data_sample.gt_instance_labels
        gt_fields = data_sample.get('gt_fields', dict())
        num_gts = len(gt_instances)

        if num_gts == 0:
            cls_target = cls_scores.new_zeros((0, self.num_classes))
            bbox_target = cls_scores.new_zeros((0, 4))
            obj_target = cls_scores.new_zeros((num_priors, 1))
            obj_weight = cls_scores.new_ones((num_priors, 1))
            kpt_target = cls_scores.new_zeros((0, self.num_keypoints, 2))
            vis_target = cls_scores.new_zeros((0, self.num_keypoints))
            vis_weight = cls_scores.new_zeros((0, self.num_keypoints))
            pos_areas = cls_scores.new_zeros((0,))
            pos_priors = priors[:0]
            foreground_mask = cls_scores.new_zeros(num_priors).bool()
            type_target = cls_scores.new_zeros((0, self.num_keypoints), dtype=torch.long)

            return (foreground_mask, cls_target, obj_target, obj_weight,
                    bbox_target, kpt_target, vis_target, vis_weight, pos_areas,
                    pos_priors, [], 0, type_target)

        scores = cls_scores * objectness
        pred_instances = InstanceData(
            bboxes=decoded_bboxes,
            scores=scores.sqrt_(),
            priors=priors,
            keypoints=decoded_kpts,
            keypoints_visible=kpt_vis,
        )
        assign_result = self.assigner.assign(
            pred_instances=pred_instances, gt_instances=gt_instances)

        pos_inds = torch.nonzero(
            assign_result['gt_inds'] > 0, as_tuple=False).squeeze(-1).unique()
        num_pos_per_img = pos_inds.size(0)
        pos_gt_labels = assign_result['labels'][pos_inds]
        pos_assigned_gt_inds = assign_result['gt_inds'][pos_inds] - 1

        bbox_target = gt_instances.bboxes[pos_assigned_gt_inds.long()]
        max_overlaps = assign_result['max_overlaps'][pos_inds]
        cls_target = F.one_hot(pos_gt_labels, self.num_classes) * max_overlaps.unsqueeze(-1)

        kpt_target = gt_instances.keypoints[pos_assigned_gt_inds]
        vis_encoded = gt_instances.keypoints_visible[pos_assigned_gt_inds]

        type_target = (vis_encoded // 10).long()
        vis_target = (vis_encoded % 10).float()

        if 'keypoints_visible_weights' in gt_instances:
            vis_weight = gt_instances.keypoints_visible_weights[pos_assigned_gt_inds]
        else:
            vis_weight = vis_target.new_ones(vis_target.shape)

        pos_areas = gt_instances.areas[pos_assigned_gt_inds]

        obj_target = torch.zeros_like(objectness)
        obj_target[pos_inds] = 1

        invalid_mask = gt_fields.get('heatmap_mask', None)
        if invalid_mask is not None and (invalid_mask != 0.0).any():
            pred_vis = (kpt_vis.unsqueeze(-1) > 0.3).float()
            mean_kpts = (decoded_kpts * pred_vis).sum(dim=1) / pred_vis.sum(dim=1).clamp(min=1e-8)
            mean_kpts = mean_kpts.reshape(1, -1, 1, 2)
            wh = invalid_mask.shape[-1]
            grids = mean_kpts / (wh - 1) * 2 - 1
            mask = invalid_mask.unsqueeze(0).float()
            weight = F.grid_sample(mask, grids, mode='bilinear', padding_mode='zeros')
            obj_weight = 1.0 - weight.reshape(num_priors, 1)
        else:
            obj_weight = obj_target.new_ones(obj_target.shape)

        foreground_mask = torch.zeros_like(objectness.squeeze()).to(torch.bool)
        foreground_mask[pos_inds] = 1
        pos_priors = priors[pos_inds]
        group_index = [
            torch.where(pos_assigned_gt_inds == num)[0]
            for num in torch.unique(pos_assigned_gt_inds)
        ]

        return (foreground_mask, cls_target, obj_target, obj_weight,
                bbox_target, kpt_target, vis_target, vis_weight, pos_areas,
                pos_priors, group_index, num_pos_per_img, type_target)

    def predict(self,
                feats: Features,
                batch_data_samples: OptSampleList,
                test_cfg: ConfigType = {}) -> Predictions:

        # 回归到原生的预测，剔除掉 type_preds 及其相关操作
        return super().predict(feats, batch_data_samples, test_cfg)