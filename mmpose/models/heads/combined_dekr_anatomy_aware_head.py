from typing import Tuple

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import Tensor

from mmpose.models.utils.tta import flip_heatmaps
from mmpose.registry import MODELS
from mmpose.models import DEKRHead
from mmcv.cnn import ConvModule


@MODELS.register_module()
class CombinedDEKRAnatomyAwareHead(DEKRHead):
    def __init__(self,
                 type_loss_weight=1.0,
                 tau=1.0,
                 bio_loss_weight=1.0,
                 with_contrastive=False,
                 **kwargs):
        super().__init__(**kwargs)

        # 1. 注册全局回归权重与类别权重表 (基于你的 ProPose 数据集统计)
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

        # 2. 类别预测分支
        self.type_conv_layers = nn.Sequential(
            ConvModule(self.in_channels, 256, 1, norm_cfg=dict(type='BN')),
            nn.Conv2d(256, self.num_keypoints * 3, 1)
        )

        self.tau = tau
        self.type_loss_weight = type_loss_weight
        self.bio_loss_weight = bio_loss_weight
        self.with_contrastive = with_contrastive
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')

        self.omega_dict = {
            23: [7, 9, 17, 25], 24: [8, 10, 18, 26],
            25: [9, 17, 23], 26: [10, 18, 24],
            27: [13, 15, 19, 21, 29], 28: [14, 16, 20, 22, 30],
            29: [15, 19, 21, 27], 30: [16, 20, 22, 28],
        }

    def forward(self, feats: Tuple[Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        x = feats[-1]
        heatmaps = self.heatmap_conv_layers(x)
        displacements = self.displacement_conv_layers(x)
        type_logits = self.type_conv_layers(x)
        return heatmaps, displacements, type_logits

    def loss(self, feats, batch_data_samples, train_cfg={}):
        pred_heatmaps, pred_displacements, pred_type_logits = self.forward(feats)

        gt_heatmaps = torch.stack([d.gt_fields.heatmaps for d in batch_data_samples])
        gt_displacements = torch.stack([d.gt_fields.displacements for d in batch_data_samples])
        raw_dis_weights = torch.stack([d.gt_fields.displacement_weights for d in batch_data_samples])

        B, K2, H, W = raw_dis_weights.shape
        K = self.num_keypoints

        # 🌟 解码隐藏在权重中的 type 和真正的 vis
        type_target_map = (raw_dis_weights[:, :K] // 10).long()
        vis_target_map = (raw_dis_weights[:, :K] % 10).float()

        # 🌟 空间级权重查表
        custom_reg_weights = torch.zeros_like(vis_target_map)
        for k in range(K):
            for t_val in [0, 1]:
                mask = (type_target_map[:, k] == t_val)
                custom_reg_weights[:, k][mask] = self.w_reg_table[k, t_val]

        final_dis_weights = torch.cat([vis_target_map * custom_reg_weights] * 2, dim=1)

        losses = dict()
        heatmap_weights = torch.stack([d.gt_fields.heatmap_weights for d in batch_data_samples])
        heatmap_mask = torch.stack([d.gt_fields.heatmap_mask for d in batch_data_samples]) if 'heatmap_mask' in \
                                                                                              batch_data_samples[
                                                                                                  0].gt_fields else None

        losses['loss/heatmap'] = self.loss_module['heatmap'](pred_heatmaps, gt_heatmaps, heatmap_weights, heatmap_mask)
        losses['loss/displacement'] = self.loss_module['displacement'](pred_displacements, gt_displacements,
                                                                       final_dis_weights)

        # 🌟 分类 Loss (密集计算)
        ce_mask = (vis_target_map > 0).float()
        pred_type_logits_ce = pred_type_logits.view(B, K, 3, H, W).permute(0, 2, 1, 3, 4)
        loss_type = self.ce_loss(pred_type_logits_ce, type_target_map)
        losses['loss/type'] = (loss_type * ce_mask).sum() / (ce_mask.sum() + 1e-6) * self.type_loss_weight

        # 🌟 BioContrastive Loss (适配 DEKR 的空间维度)
        if self.with_contrastive:
            type_probs = torch.softmax(pred_type_logits.view(B, K, 3, H, W), dim=2)
            p_bio = type_probs[:, :, 0]  # 正常类概率 [B, K, H, W]

            loss_bio_total, valid_r_count = 0.0, 0.0
            for r, omega_r in self.omega_dict.items():
                v_r_mask = ((vis_target_map[:, r] > 0) & (type_target_map[:, r] == 0)).float()
                exp_p_r = torch.exp(p_bio[:, r] / self.tau)

                sum_exp_j = torch.zeros_like(exp_p_r)
                for j in omega_r:
                    v_j_mask = (vis_target_map[:, j] > 0).float()
                    sum_exp_j += v_j_mask * torch.exp(p_bio[:, j] / self.tau)

                prob_r = exp_p_r / (exp_p_r + sum_exp_j + 1e-6)
                loss_bio_total += (-torch.log(prob_r + 1e-6) * v_r_mask).sum()
                valid_r_count += v_r_mask.sum()

            losses['loss/bio'] = self.bio_loss_weight * (loss_bio_total / (valid_r_count + 1e-6))

        return losses

    def predict(self, feats, batch_data_samples, test_cfg={}):
        assert len(batch_data_samples) == 1, f'DEKRHead only supports ' \
                                             f'prediction with batch_size 1, but got {len(batch_data_samples)}'

        multiscale_test = test_cfg.get('multiscale_test', False)
        flip_test = test_cfg.get('flip_test', False)
        metainfo = batch_data_samples[0].metainfo
        aug_scales = [1]

        if not multiscale_test:
            feats = [feats]
        else:
            aug_scales = aug_scales + metainfo['aug_scales']

        heatmaps, displacements = [], []
        # 🌟 声明一个变量，专门用来存提取 type_logits
        final_type_logits = None

        for feat, s in zip(feats, aug_scales):
            if flip_test:
                assert isinstance(feat, list) and len(feat) == 2
                flip_indices = metainfo['flip_indices']
                _feat, _feat_flip = feat

                # 🌟 核心修改 1：接收 3 个返回值！
                _heatmaps, _displacements, _type_logits = self.forward(_feat)
                _heatmaps_flip, _displacements_flip, _ = self.forward(_feat_flip)

                # 翻转融合逻辑 (与官方完全一致)
                _heatmaps_flip = flip_heatmaps(
                    _heatmaps_flip,
                    flip_mode='heatmap',
                    flip_indices=flip_indices + [len(flip_indices)],
                    shift_heatmap=test_cfg.get('shift_heatmap', False))
                _heatmaps = (_heatmaps + _heatmaps_flip) / 2.0

                _displacements_flip = flip_heatmaps(
                    _displacements_flip,
                    flip_mode='offset',
                    flip_indices=flip_indices,
                    shift_heatmap=False)

                x_scale_factor = s * (metainfo['input_size'][0] / _heatmaps.shape[-1])
                _displacements_flip[:, ::2] += (x_scale_factor - 1) / x_scale_factor
                _displacements = (_displacements + _displacements_flip) / 2.0

                # 🌟 为了不增加翻转类别预测的复杂度，分类我们直接使用原图(未翻转)的特征
                final_type_logits = _type_logits

            else:
                # 🌟 未开启 flip_test 的正常流程，同样接收 3 个返回值
                _heatmaps, _displacements, final_type_logits = self.forward(feat)

            # 存入列表，完美契合 decode 的期望格式！
            heatmaps.append(_heatmaps)
            displacements.append(_displacements)

        # 调用官方底层解码器，因为传的是 List，就不会报错了
        preds = self.decode(heatmaps, displacements, test_cfg, metainfo)

        # -----------------------------------------------------------------
        # 🌟 下面完全是我们自己的 Grid Sample 提取类别的逻辑 (一行没动)
        # -----------------------------------------------------------------
        B, C, H, W = final_type_logits.shape
        type_logits_view = final_type_logits.view(B, self.num_keypoints, 3, H, W)

        for i, results in enumerate(preds):
            if len(results.keypoints) > 0:
                kpts = torch.from_numpy(results.keypoints).to(final_type_logits.device)

                # 归一化坐标到 [-1, 1] 用于 grid_sample
                img_w, img_h = batch_data_samples[i].metainfo['input_size']
                grid_kpts = kpts.clone()
                grid_kpts[..., 0] = (grid_kpts[..., 0] / (img_w - 1)) * 2 - 1
                grid_kpts[..., 1] = (grid_kpts[..., 1] / (img_h - 1)) * 2 - 1

                N = kpts.shape[0]
                sampled_types = []
                for k in range(self.num_keypoints):
                    single_kpt_logits = type_logits_view[i:i + 1, k]
                    single_grid = grid_kpts[:, k:k + 1, :].view(1, N, 1, 2)
                    sampled = F.grid_sample(single_kpt_logits, single_grid, align_corners=True)
                    sampled_types.append(sampled.view(3, N).T)

                all_sampled_logits = torch.stack(sampled_types, dim=1)
                type_probs = torch.softmax(all_sampled_logits, dim=-1)
                pred_types = torch.argmax(type_probs, dim=-1)

                results.keypoint_types = pred_types.cpu().numpy()
                results.type_scores = type_probs.cpu().numpy()

        return preds