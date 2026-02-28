import copy
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import ConvModule
from mmengine.structures import InstanceData
from torch import Tensor

from mmpose.models.heads.hybrid_heads import YOLOXPoseHead
from mmpose.models.heads.hybrid_heads.yoloxpose_head import YOLOXPoseHeadModule
from mmpose.registry import MODELS
from mmpose.structures import PoseDataSample
from mmpose.utils import reduce_mean
from mmpose.utils.typing import (ConfigType, Features, OptSampleList,
                                 Predictions, SampleList)
from mmpose.models.utils import filter_scores_and_topk
from mmpose.evaluation.functional import nms_torch


@MODELS.register_module()
class YOLOAnatomyAwareHeadModule(YOLOXPoseHeadModule):
    def _init_layers(self):
        super()._init_layers()
        self.out_type = nn.ModuleList()
        for _ in self.featmap_strides:
            self.out_type.append(nn.Conv2d(self.feat_channels, self.num_keypoints * 3, 1))

    def init_weights(self):
        super().init_weights()
        for conv_type in self.out_type:
            nn.init.normal_(conv_type.weight, mean=0, std=0.01)
            if conv_type.bias is not None:
                nn.init.constant_(conv_type.bias, 0)

    def forward(self, x):
        """重写前向传播，把 5 个输出变成 6 个"""
        cls_scores, objectnesses, bbox_preds = [], [], []
        kpt_offsets, kpt_vis, type_preds = [], [], []

        for i in range(len(x)):
            # 走原生的特征提取
            cls_feat = self.conv_cls[i](x[i])
            reg_feat = self.conv_reg[i](x[i])
            pose_feat = self.conv_pose[i](x[i])

            cls_scores.append(self.out_cls[i](cls_feat))
            objectnesses.append(self.out_obj[i](reg_feat))
            bbox_preds.append(self.out_bbox[i](reg_feat))
            kpt_offsets.append(self.out_kpt[i](pose_feat))
            kpt_vis.append(self.out_kpt_vis[i](pose_feat))
            type_preds.append(self.out_type[i](pose_feat))

        return cls_scores, objectnesses, bbox_preds, kpt_offsets, kpt_vis, type_preds

@MODELS.register_module()
class CombinedYOLOAnatomyAwareHead(YOLOXPoseHead):
    def __init__(self,
                 type_loss_weight=1.0,
                 tau=1.0,
                 bio_loss_weight=1.0,
                 with_contrastive=False,
                 **kwargs):
        # 初始化原生的 YOLOXPoseHead (包含 bbox, kpt 分支)
        head_module_cfg = kwargs.get('head_module_cfg', dict()).copy()
        super().__init__(**kwargs)

        self.register_buffer('w_reg_table', torch.tensor([
            [0.9348983, 0.0], [0.9356371, 0.0], [0.93539065, 0.0],
            [0.93615663, 0.0], [0.9361826, 0.0], [0.9283144, 0.0],
            [0.9279217, 0.0], [0.98167354, 5.0], [0.960146, 5.0],
            [1.1028655, 3.6270287], [1.0354266, 3.7909665], [0.9303362, 0.0],
            [0.93055314, 0.0], [1.0990598, 3.0702474], [1.0660323, 3.319553],
            [1.2535793, 2.9189765], [1.2353523, 2.9720054], [1.11805, 2.9624858],
            [1.052587, 3.0962512], [1.2955451, 2.2605414], [1.2715614, 2.1721284],
            [1.3000122, 2.2717884], [1.2757512, 2.187222], [3.3312047, 0.0],
            [4.2261786, 0.0], [2.2803817, 0.0], [2.7141972, 0.0],
            [2.5414956, 0.0], [2.866074, 0.0], [3.1917245, 0.0],
            [2.7873797, 0.0]
        ], dtype=torch.float32))

        # 2. 注册类别分类权重 [31, 3] (如果不为空)
        self.register_buffer('w_type_table', torch.tensor([
            [1.0000, 0.0000, 0.0000], [1.0000, 0.0000, 0.0000], [1.0000, 0.0000, 0.0000],
            [1.0000, 0.0000, 0.0000], [1.0000, 0.0000, 0.0000], [1.0000, 0.0000, 0.0000],
            [1.0000, 0.0000, 0.0000], [0.2738, 1.6923, 1.0339], [0.2336, 1.4522, 1.3141],
            [0.4824, 1.5864, 0.9312], [0.4103, 1.5021, 1.0876], [1.0000, 0.0000, 0.0000],
            [1.0000, 0.0000, 0.0000], [0.4103, 1.1461, 1.4437], [0.3303, 1.0286, 1.6411],
            [0.5608, 1.3059, 1.1333], [0.5414, 1.3025, 1.1561], [0.5213, 1.3813, 1.0974],
            [0.4298, 1.2644, 1.3057], [0.5398, 0.9419, 1.5183], [0.4575, 0.7814, 1.7611],
            [0.5404, 0.9444, 1.5151], [0.4577, 0.7847, 1.7576], [1.5448, 0.0000, 0.4552],
            [1.6297, 0.0000, 0.3703], [1.3732, 0.0000, 0.6268], [1.4603, 0.0000, 0.5397],
            [1.3962, 0.0000, 0.6038], [1.4578, 0.0000, 0.5422], [1.4756, 0.0000, 0.5244],
            [1.4144, 0.0000, 0.5856]
        ], dtype=torch.float32))

        self.tau = tau
        self.type_loss_weight = type_loss_weight
        self.bio_loss_weight = bio_loss_weight
        self.with_contrastive = with_contrastive
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        if head_module_cfg:
            head_module_cfg['featmap_strides'] = self.featmap_strides
            head_module_cfg['num_keypoints'] = self.num_keypoints
            self.head_module = YOLOAnatomyAwareHeadModule(**head_module_cfg)

        # 解剖学拓扑字典
        self.omega_dict = {
            23: [7, 9, 17, 25], 24: [8, 10, 18, 26],
            25: [9, 17, 23], 26: [10, 18, 24],
            27: [13, 15, 19, 21, 29], 28: [14, 16, 20, 22, 30],
            29: [15, 19, 21, 27], 30: [16, 20, 22, 28],
        }

    def forward(self, feats: Features, with_type=False):
        if with_type:
            return self.head_module(feats)
        cls_scores, objectnesses, bbox_preds, kpt_offsets, kpt_vis, _ = self.head_module(feats)
        return cls_scores, objectnesses, bbox_preds, kpt_offsets, kpt_vis

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
            kpt_vis, type_preds = self.forward(feats, with_type=True)

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
        flatten_type_preds = self._flatten_predictions(type_preds)  # [B*N_anchors, K*3]

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
            num_fg_imgs, type_targets, custom_weight_targets = targets

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

            weight = vis_targets * custom_weight_targets
            losses['loss_kpt'] = self.loss_oks(kpt_preds, kpt_targets,
                                               weight, pos_areas)

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

            pos_type_logits = flatten_type_preds.view(-1, self.num_keypoints, 3)[pos_masks]
            N_pos = pos_type_logits.shape[0]
            global_type_weights = self.w_type_table

            expanded_type_weights = global_type_weights.unsqueeze(0).expand(N_pos, -1, -1)
            gathered_type_weights = expanded_type_weights.gather(
                dim=2, index=type_targets.unsqueeze(2).long()
            ).squeeze(-1)
            ce_mask = (vis_targets > 0).float() * gathered_type_weights
            raw_loss_type = self.ce_loss(pos_type_logits.view(-1, 3), type_targets.view(-1).long())
            loss_type = (raw_loss_type * ce_mask.view(-1)).sum() / (ce_mask.sum() + 1e-6)
            losses['loss_type'] = self.type_loss_weight * loss_type

            # 4. BioContrastive Loss (完美复刻你原本的逻辑！)
            if self.with_contrastive:
                type_probs = torch.softmax(pos_type_logits, dim=-1)
                p_bio = type_probs[:, :, 0]

                loss_bio_total = 0.0
                valid_r_count = 0.0

                for r, omega_r in self.omega_dict.items():
                    # 只有当该点真实存在且被标为"正常(0)"时，才算作有效残肢起点
                    v_r_mask = ((vis_targets[:, r] > 0) & (type_targets[:, r] == 0)).float()

                    exp_p_r = torch.exp(p_bio[:, r] / self.tau)

                    sum_exp_j = torch.zeros(N_pos, device=flatten_cls_scores.device)
                    for j in omega_r:
                        v_j_mask = (vis_targets[:, j] > 0).float()
                        sum_exp_j += v_j_mask * torch.exp(p_bio[:, j] / self.tau)

                    prob_r = exp_p_r / (exp_p_r + sum_exp_j + 1e-6)
                    loss_r = -torch.log(prob_r + 1e-6)

                    loss_bio_total += (loss_r * v_r_mask).sum()
                    valid_r_count += v_r_mask.sum()

                if valid_r_count > 0:
                    losses['loss_bio'] = self.bio_loss_weight * (loss_bio_total / valid_r_count)
                else:
                    losses['loss_bio'] = p_bio.sum() * 0.0

            # 5. Type Accuracy (供训练面板打印)
            with torch.no_grad():
                pred_classes = torch.argmax(pos_type_logits, dim=-1)
                correct = (pred_classes == type_targets)
                acc_type = (correct.float() * (vis_targets > 0).float()).sum() / (
                            (vis_targets > 0).float().sum() + 1e-6)
                losses['acc_type'] = acc_type

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

        # use clip to avoid nan
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
            pos_priors, group_indices, num_pos_per_img, type_targets, custom_weight_targets = targets

        # post-processing for targets
        if self.use_aux_loss:
            bbox_cxcy = (bbox_targets[:, :2] + bbox_targets[:, 2:]) / 2.0
            bbox_wh = bbox_targets[:, 2:] - bbox_targets[:, :2]
            bbox_aux_targets = torch.cat([
                (bbox_cxcy - pos_priors[:, :2]) / pos_priors[:, 2:],
                torch.log(bbox_wh / pos_priors[:, 2:] + 1e-8)
            ],
                dim=-1)

            kpt_aux_targets = (kpt_targets - pos_priors[:, None, :2]) \
                              / pos_priors[:, None, 2:]
        else:
            bbox_aux_targets, kpt_aux_targets = None, None

        return (foreground_masks, cls_targets, obj_targets, obj_weights,
                bbox_targets, bbox_aux_targets, kpt_targets, kpt_aux_targets,
                vis_targets, vis_weights, pos_areas, pos_priors, group_indices,
                num_pos_per_img, type_targets, custom_weight_targets)

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
        """Compute classification, bbox, keypoints and objectness targets for
        priors in a single image.

        Args:
            priors (Tensor): All priors of one image, a 2D-Tensor with shape
                [num_priors, 4] in [cx, xy, stride_w, stride_y] format.
            cls_scores (Tensor): Classification predictions of one image,
                a 2D-Tensor with shape [num_priors, num_classes]
            objectness (Tensor): Objectness predictions of one image,
                a 1D-Tensor with shape [num_priors]
            decoded_bboxes (Tensor): Decoded bboxes predictions of one image,
                a 2D-Tensor with shape [num_priors, 4] in xyxy format.
            decoded_kpts (Tensor): Decoded keypoints predictions of one image,
                a 3D-Tensor with shape [num_priors, num_keypoints, 2].
            kpt_vis (Tensor): Keypoints visibility predictions of one image,
                a 2D-Tensor with shape [num_priors, num_keypoints].
            gt_instances (:obj:`InstanceData`): Ground truth of instance
                annotations. It should includes ``bboxes`` and ``labels``
                attributes.
            data_sample (PoseDataSample): Data sample that contains the ground
                truth annotations for current image.

        Returns:
            tuple: A tuple containing various target tensors for training:
                - foreground_mask (Tensor): Binary mask indicating foreground
                    priors.
                - cls_target (Tensor): Classification targets.
                - obj_target (Tensor): Objectness targets.
                - obj_weight (Tensor): Weights for objectness targets.
                - bbox_target (Tensor): BBox targets.
                - kpt_target (Tensor): Keypoints targets.
                - vis_target (Tensor): Visibility targets for keypoints.
                - vis_weight (Tensor): Weights for keypoints visibility
                    targets.
                - pos_areas (Tensor): Areas of positive samples.
                - pos_priors (Tensor): Priors corresponding to positive
                    samples.
                - group_index (List[Tensor]): Indices of groups for positive
                    samples.
                - num_pos_per_img (int): Number of positive samples.
        """
        # TODO: change the shape of objectness to [num_priors]
        num_priors = priors.size(0)
        gt_instances = data_sample.gt_instance_labels
        gt_fields = data_sample.get('gt_fields', dict())
        num_gts = len(gt_instances)

        # No target
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
            custom_weight_target = cls_scores.new_zeros((0, self.num_keypoints))
            return (foreground_mask, cls_target, obj_target, obj_weight,
                    bbox_target, kpt_target, vis_target, vis_weight, pos_areas,
                    pos_priors, [], 0, type_target, custom_weight_target)

        # assign positive samples
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

        # sampling
        pos_inds = torch.nonzero(
            assign_result['gt_inds'] > 0, as_tuple=False).squeeze(-1).unique()
        num_pos_per_img = pos_inds.size(0)
        pos_gt_labels = assign_result['labels'][pos_inds]
        pos_assigned_gt_inds = assign_result['gt_inds'][pos_inds] - 1

        # bbox target
        bbox_target = gt_instances.bboxes[pos_assigned_gt_inds.long()]

        # cls target
        max_overlaps = assign_result['max_overlaps'][pos_inds]
        cls_target = F.one_hot(pos_gt_labels,
                               self.num_classes) * max_overlaps.unsqueeze(-1)

        # pose targets
        kpt_target = gt_instances.keypoints[pos_assigned_gt_inds]
        vis_encoded = gt_instances.keypoints_visible[pos_assigned_gt_inds]
        type_target = (vis_encoded // 10).long()
        vis_target = (vis_encoded % 10).float()

        custom_weight_target = vis_target.new_zeros(vis_target.shape)
        for k in range(self.num_keypoints):
            t = type_target[:, k]

            # Type 0 (正常) 和 Type 1 (假肢) 去表里查权重
            valid_mask = (t == 0) | (t == 1)
            if valid_mask.any():
                # 拿着 k 和 t 去刚才存好的 self.w_reg_table 里面查
                custom_weight_target[valid_mask, k] = self.w_reg_table[k, t[valid_mask]]

        if 'keypoints_visible_weights' in gt_instances:
            vis_weight = gt_instances.keypoints_visible_weights[
                pos_assigned_gt_inds]
        else:
            vis_weight = vis_target.new_ones(vis_target.shape)
        pos_areas = gt_instances.areas[pos_assigned_gt_inds]

        # obj target
        obj_target = torch.zeros_like(objectness)
        obj_target[pos_inds] = 1

        invalid_mask = gt_fields.get('heatmap_mask', None)
        if invalid_mask is not None and (invalid_mask != 0.0).any():
            # ignore the tokens that predict the unlabled instances
            pred_vis = (kpt_vis.unsqueeze(-1) > 0.3).float()
            mean_kpts = (decoded_kpts * pred_vis).sum(dim=1) / pred_vis.sum(
                dim=1).clamp(min=1e-8)
            mean_kpts = mean_kpts.reshape(1, -1, 1, 2)
            wh = invalid_mask.shape[-1]
            grids = mean_kpts / (wh - 1) * 2 - 1
            mask = invalid_mask.unsqueeze(0).float()
            weight = F.grid_sample(
                mask, grids, mode='bilinear', padding_mode='zeros')
            obj_weight = 1.0 - weight.reshape(num_priors, 1)
        else:
            obj_weight = obj_target.new_ones(obj_target.shape)

        # misc
        foreground_mask = torch.zeros_like(objectness.squeeze()).to(torch.bool)
        foreground_mask[pos_inds] = 1
        pos_priors = priors[pos_inds]
        group_index = [
            torch.where(pos_assigned_gt_inds == num)[0]
            for num in torch.unique(pos_assigned_gt_inds)
        ]

        return (foreground_mask, cls_target, obj_target, obj_weight,
                bbox_target, kpt_target, vis_target, vis_weight, pos_areas,
                pos_priors, group_index, num_pos_per_img, type_target, custom_weight_target)

    def predict(self,
                feats: Features,
                batch_data_samples: OptSampleList,
                test_cfg: ConfigType = {}) -> Predictions:

        cls_scores, objectnesses, bbox_preds, kpt_offsets, \
            kpt_vis, type_preds = self.forward(feats, with_type=True)

        cfg = copy.deepcopy(test_cfg)

        batch_img_metas = [d.metainfo for d in batch_data_samples]
        featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]

        # If the shape does not change, use the previous mlvl_priors
        if featmap_sizes != self.featmap_sizes:
            self.mlvl_priors = self.prior_generator.grid_priors(
                featmap_sizes,
                dtype=cls_scores[0].dtype,
                device=cls_scores[0].device)
            self.featmap_sizes = featmap_sizes
        flatten_priors = torch.cat(self.mlvl_priors)

        mlvl_strides = [
            flatten_priors.new_full((featmap_size.numel(), ),
                                    stride) for featmap_size, stride in zip(
                                        featmap_sizes, self.featmap_strides)
        ]
        flatten_stride = torch.cat(mlvl_strides)

        # flatten cls_scores, bbox_preds and objectness
        flatten_cls_scores = self._flatten_predictions(cls_scores).sigmoid()
        flatten_bbox_preds = self._flatten_predictions(bbox_preds)
        flatten_objectness = self._flatten_predictions(objectnesses).sigmoid()
        flatten_kpt_offsets = self._flatten_predictions(kpt_offsets)
        flatten_kpt_vis = self._flatten_predictions(kpt_vis).sigmoid()
        flatten_bbox_preds = self.decode_bbox(flatten_bbox_preds,
                                              flatten_priors, flatten_stride)
        flatten_kpt_reg = self.decode_kpt_reg(flatten_kpt_offsets,
                                              flatten_priors, flatten_stride)
        flatten_type_preds = self._flatten_predictions(type_preds)

        results_list = []
        for (bboxes, scores, objectness, kpt_reg, kpt_vis, type_pred,
             img_meta) in zip(flatten_bbox_preds, flatten_cls_scores,
                              flatten_objectness, flatten_kpt_reg,
                              flatten_kpt_vis, flatten_type_preds, batch_img_metas):

            score_thr = cfg.get('score_thr', 0.01)
            scores *= objectness

            nms_pre = cfg.get('nms_pre', 100000)
            scores, labels = scores.max(1, keepdim=True)
            scores, _, keep_idxs_score, results = filter_scores_and_topk(
                scores, score_thr, nms_pre, results=dict(labels=labels[:, 0]))
            labels = results['labels']

            bboxes = bboxes[keep_idxs_score]
            kpt_vis = kpt_vis[keep_idxs_score]
            stride = flatten_stride[keep_idxs_score]
            keypoints = kpt_reg[keep_idxs_score]
            type_pred = type_pred[keep_idxs_score]

            if bboxes.numel() > 0:
                nms_thr = cfg.get('nms_thr', 1.0)
                if nms_thr < 1.0:
                    keep_idxs_nms = nms_torch(bboxes, scores, nms_thr)
                    bboxes = bboxes[keep_idxs_nms]
                    stride = stride[keep_idxs_nms]
                    labels = labels[keep_idxs_nms]
                    kpt_vis = kpt_vis[keep_idxs_nms]
                    keypoints = keypoints[keep_idxs_nms]
                    scores = scores[keep_idxs_nms]
                    type_pred = type_pred[keep_idxs_nms]

            type_pred = type_pred.view(-1, self.num_keypoints, 3)
            type_probs = torch.softmax(type_pred, dim=-1)
            pred_types = torch.argmax(type_probs, dim=-1)

            results = InstanceData(
                scores=scores,
                labels=labels,
                bboxes=bboxes,
                bbox_scores=scores,
                keypoints=keypoints,
                keypoint_scores=kpt_vis,
                keypoints_visible=kpt_vis,
                keypoint_types=pred_types.cpu().numpy(),
                type_scores=type_probs.cpu().numpy(),
            )

            input_size = img_meta['input_size']
            results.bboxes[:, 0::2].clamp_(0, input_size[0])
            results.bboxes[:, 1::2].clamp_(0, input_size[1])

            results_list.append(results.numpy())

        return results_list
