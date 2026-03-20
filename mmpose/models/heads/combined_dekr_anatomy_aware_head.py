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

        # 🌟 1. 创建四张空白画布：Type, Vis, Reg Weight, Type Weight
        type_target_map = torch.zeros((B, K, H, W), dtype=torch.long, device=pred_heatmaps.device)
        vis_target_map = torch.zeros((B, K, H, W), dtype=torch.float32, device=pred_heatmaps.device)
        reg_weight_map = torch.ones((B, K, H, W), dtype=torch.float32, device=pred_heatmaps.device)
        type_weight_map = torch.zeros((B, K, H, W), dtype=torch.float32, device=pred_heatmaps.device)

        # 🌟 2. 提取全局类别权重并解包 (只需要拿 batch[0] 的即可，因为是全局一致的)
        g_weights = batch_data_samples[0].gt_instances.global_type_weights
        if isinstance(g_weights, list): g_weights = g_weights[0]
        if hasattr(g_weights, 'cpu'): g_weights = g_weights.cpu().numpy()
        g_weights = g_weights.squeeze()  # 降维成纯净的 [K, 3] 矩阵

        # 🌟 3. 遍历 Batch，开始当“粉刷匠”
        for b in range(B):
            img_w, img_h = batch_data_samples[b].metainfo['input_size']
            scale_w, scale_h = W / img_w, H / img_h

            gt_instances = batch_data_samples[b].gt_instances
            if 'keypoints' not in gt_instances:
                continue

            kpts = gt_instances.keypoints
            vis_real = gt_instances.keypoints_visible
            N = kpts.shape[0]
            # 安全解包 Types
            types = gt_instances.keypoint_types
            if isinstance(types, list):
                if len(types) == N:  # 如果是 [Tensor(K), Tensor(K)] 这种按人分开的格式
                    types = torch.stack(types) if hasattr(types[0], 'cpu') else np.stack(types)
                else:  # 如果是 [Tensor(N, K)] 这种整体包一层的格式
                    raise ValueError('types lengthe is not equal to N')
            if hasattr(types, 'cpu'):
                types = types.cpu().numpy()
            types = np.reshape(types, (N, K))  # 强制拉平，杜绝一切后患

            # 安全解包 Custom Reg Weights
            inst_weights = gt_instances.custom_reg_weights
            if isinstance(inst_weights, list): inst_weights = inst_weights[0]
            if hasattr(inst_weights, 'cpu'): inst_weights = inst_weights.cpu().numpy()

            for n in range(kpts.shape[0]):
                for k in range(K):
                    if vis_real[n, k] > 0:
                        cx = int(kpts[n, k, 0] * scale_w)
                        cy = int(kpts[n, k, 1] * scale_h)

                        r = 3
                        y_min, y_max = max(0, cy - r), min(H, cy + r + 1)
                        x_min, x_max = max(0, cx - r), min(W, cx + r + 1)

                        if y_min < y_max and x_min < x_max:
                            t_val = types[n, k]  # 当前点的类别

                            # 🌟 四笔齐下：把类别、可见度、回归权重、分类权重，精准涂抹在对应像素块上！
                            type_target_map[b, k, y_min:y_max, x_min:x_max] = t_val
                            vis_target_map[b, k, y_min:y_max, x_min:x_max] = 1.0
                            reg_weight_map[b, k, y_min:y_max, x_min:x_max] = float(inst_weights[n, k, 0])
                            type_weight_map[b, k, y_min:y_max, x_min:x_max] = float(g_weights[k, t_val])

        # 🌟 4. 完美融合底层的高斯掩码与我们的回归权重
        raw_dis_k = raw_dis_weights[:, :K] * reg_weight_map
        final_dis_weights = torch.cat([raw_dis_k, raw_dis_k], dim=1)

        # ---------------------------------------------------------
        # 🌟 5. 开始计算各项 Loss
        # ---------------------------------------------------------
        losses = dict()

        heatmap_weights = torch.stack([d.gt_fields.heatmap_weights for d in batch_data_samples])
        if 'heatmap_mask' in batch_data_samples[0].gt_fields.keys():
            heatmap_mask = torch.stack([d.gt_fields.heatmap_mask for d in batch_data_samples])
        else:
            heatmap_mask = None

        losses['loss/heatmap'] = self.loss_module['heatmap'](
            pred_heatmaps, gt_heatmaps, heatmap_weights, heatmap_mask)
        losses['loss/displacement'] = self.loss_module['displacement'](
            pred_displacements, gt_displacements, final_dis_weights)

        # 🌟 分类 Type Loss (完美对齐 ViT 的 gather 加权逻辑)
        base_ce_mask = (raw_dis_weights[:, :K] > 0).float()
        ce_mask = base_ce_mask * type_weight_map  # 乘上查表画好的类别权重

        pred_type_logits_ce = pred_type_logits.view(B, K, 3, H, W).permute(0, 2, 1, 3, 4)
        raw_loss_type = self.ce_loss(pred_type_logits_ce, type_target_map)

        loss_type = (raw_loss_type * ce_mask).sum() / (ce_mask.sum() + 1e-6)
        losses['loss/type'] = self.type_loss_weight * loss_type

        # 🌟 BioContrastive Loss (生物力学对比损失)
        if self.with_contrastive:
            type_probs = torch.softmax(pred_type_logits.view(B, K, 3, H, W), dim=2)
            p_bio = type_probs[:, :, 0]  # 取正常类别的概率图 [B, K, H, W]

            loss_bio_total, valid_r_count = 0.0, 0.0
            for r, omega_r in self.omega_dict.items():
                # 🌟 注意这里的 mask 逻辑：r 是残肢/假肢点，所以 type 应该 > 0 (1代表假肢, 2代表残肢)
                # 你之前的代码写的是 == 0，那是正常点，逻辑反了
                v_r_mask = ((vis_target_map[:, r] > 0) & (type_target_map[:, r] > 0)).float()

                # 计算有效像素个数
                r_counts = v_r_mask.sum(dim=(1, 2))  # [B]

                # 🌟 只在有效区域求均值，而不是全图均值
                exp_p_r = (torch.exp(p_bio[:, r] / self.tau) * v_r_mask).sum(dim=(1, 2)) / (r_counts + 1e-6)

                sum_exp_j = torch.zeros_like(exp_p_r)  # [B]
                for j in omega_r:
                    v_j_mask = (vis_target_map[:, j] > 0).float()
                    j_counts = v_j_mask.sum(dim=(1, 2))

                    # 同理，计算参考点 j 的区域平均能量
                    energy_j = (torch.exp(p_bio[:, j] / self.tau) * v_j_mask).sum(dim=(1, 2)) / (j_counts + 1e-6)
                    sum_exp_j += energy_j

                # 只有当残肢点 r 存在时才计算
                valid_mask = (r_counts > 0).float()

                prob_r = exp_p_r / (exp_p_r + sum_exp_j + 1e-6)
                # 🌟 这里已经是 [B] 维度的标量了，直接求和即可
                loss_bio_total += (-torch.log(prob_r + 1e-6) * valid_mask).sum()
                valid_r_count += valid_mask.sum()

            losses['loss/bio'] = self.bio_loss_weight * (loss_bio_total / (valid_r_count + 1e-6))

        return losses

    def predict(self, feats, batch_data_samples, test_cfg={}):
        assert len(
            batch_data_samples) == 1, f'DEKRHead only supports prediction with batch_size 1, but got {len(batch_data_samples)}'

        multiscale_test = test_cfg.get('multiscale_test', False)
        flip_test = test_cfg.get('flip_test', False)
        metainfo = batch_data_samples[0].metainfo
        aug_scales = [1] + metainfo.get('aug_scales', []) if multiscale_test else [1]

        heatmaps, displacements = [], []
        final_type_logits = None

        for feat, s in zip(feats if multiscale_test else [feats], aug_scales):
            if flip_test:
                assert isinstance(feat, list) and len(feat) == 2
                flip_indices = metainfo['flip_indices']
                _feat, _feat_flip = feat

                _heatmaps, _displacements, _type_logits = self.forward(_feat)
                _heatmaps_flip, _displacements_flip, _ = self.forward(_feat_flip)

                _heatmaps_flip = flip_heatmaps(
                    _heatmaps_flip, flip_mode='heatmap',
                    flip_indices=flip_indices + [len(flip_indices)],
                    shift_heatmap=test_cfg.get('shift_heatmap', False))
                _heatmaps = (_heatmaps + _heatmaps_flip) / 2.0

                _displacements_flip = flip_heatmaps(
                    _displacements_flip, flip_mode='offset',
                    flip_indices=flip_indices, shift_heatmap=False)

                x_scale_factor = s * (metainfo['input_size'][0] / _heatmaps.shape[-1])
                _displacements_flip[:, ::2] += (x_scale_factor - 1) / x_scale_factor
                _displacements = (_displacements + _displacements_flip) / 2.0

                final_type_logits = _type_logits
            else:
                _heatmaps, _displacements, final_type_logits = self.forward(feat)

            heatmaps.append(_heatmaps)
            displacements.append(_displacements)

        preds = self.decode(heatmaps, displacements, test_cfg, metainfo)

        # -----------------------------------------------------------------
        # 🌟 【关键修复 3】：防止坐标时空越界导致 grid_sample 提取出纯 0 (类别 0)
        # -----------------------------------------------------------------
        B, C, H, W = final_type_logits.shape
        type_logits_view = final_type_logits.view(B, self.num_keypoints, 3, H, W)

        for i, results in enumerate(preds):
            if len(results.keypoints) > 0:
                kpts = torch.from_numpy(results.keypoints).to(final_type_logits.device)

                img_w, img_h = batch_data_samples[i].metainfo['input_size']
                ori_shape = batch_data_samples[i].metainfo['ori_shape']
                ori_h, ori_w = ori_shape[0], ori_shape[1]

                # MMPose Bottom-up Resize 通常是等比例缩放，左上角对齐(即不需要处理 pad 偏移)
                scale = min(img_w / ori_w, img_h / ori_h)

                # 把解码后的原图坐标，还原回 input_size (比如 512x512) 坐标系
                grid_kpts = kpts.clone()
                grid_kpts[..., 0] = grid_kpts[..., 0] * scale
                grid_kpts[..., 1] = grid_kpts[..., 1] * scale

                # 归一化到 [-1, 1] 供 grid_sample 使用
                grid_kpts[..., 0] = (grid_kpts[..., 0] / (img_w - 1)) * 2 - 1
                grid_kpts[..., 1] = (grid_kpts[..., 1] / (img_h - 1)) * 2 - 1

                N = kpts.shape[0]
                sampled_types = []
                for k in range(self.num_keypoints):
                    single_kpt_logits = type_logits_view[i:i + 1, k]
                    single_grid = grid_kpts[:, k:k + 1, :].view(1, N, 1, 2)
                    # align_corners=True 防止边界像素偏移
                    sampled = F.grid_sample(single_kpt_logits, single_grid, align_corners=True)
                    sampled_types.append(sampled.view(3, N).T)

                all_sampled_logits = torch.stack(sampled_types, dim=1)
                type_probs = torch.softmax(all_sampled_logits, dim=-1)
                pred_types = torch.argmax(type_probs, dim=-1)

                results.keypoint_types = pred_types.cpu().numpy()
                results.type_scores = type_probs.cpu().numpy()

        return preds