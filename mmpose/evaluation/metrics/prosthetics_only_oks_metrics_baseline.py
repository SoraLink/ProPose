# mmpose/evaluation/metrics/ldpose_metric.py
import copy
import numpy as np
from mmpose.registry import METRICS
from mmpose.evaluation.metrics import CocoMetric


@METRICS.register_module()
class ProstheticsOKSMetric(CocoMetric):
    """
    LDPose Metric: Anatomy-Aware & Type-Aware Evaluation.

    核心功能：
    1. Type-Aware Penalty (类型惩罚): 
       如果预测的材质类型 (Prosthetic/Normal) 和 GT 不一致，惩罚该点分数。

    2. Anatomy Chain Penalty (解剖连坐):
       如果 GT 显示是截肢（Root存在，Child缺失），但模型幻视出了 Child，
       则连坐惩罚 Root (残肢点) 的分数，使其在 AP 计算中失效。

    3. Ghost Rate Calculation (幻觉率):
       额外统计并输出幻觉率。
    """

    def __init__(self,
                 **kwargs):
        super().__init__(**kwargs)


    def process(self, data_batch, data_samples):
        """
        处理每个 Batch 的预测结果。
        这里我们需要确保把 'keypoint_types' 也存下来，因为父类 CocoMetric 默认可能只存坐标。
        """
        # 调用父类处理标准坐标和分数
        super().process(data_batch, data_samples)
        batch_len = len(data_samples)
        start_idx = len(self.results) - batch_len

        # 额外存储 Type 预测和 GT，用于后续 evaluate 阶段的惩罚计算
        for i, data_sample in enumerate(data_samples):
            # 存预测 Type

            target_result = self.results[start_idx + i]

            if 'keypoint_types' in data_sample['pred_instances']:
                pred_type = data_sample['pred_instances']['keypoint_types'].cpu().numpy()
            else:
                num_kps = data_sample['pred_instances']['keypoints'].shape[1]
                # 如果没有 Type Head，默认全 0 (Normal)
                pred_type = np.zeros((num_kps,), dtype=int)

            # 存 GT Type
            gt_type = data_sample['gt_instances']['keypoint_types'].cpu().numpy()
            #############################
            types = np.array(gt_type, dtype=np.int64)
            _, absent_indices = np.where(types == 2)
            if len(absent_indices) > 0:
                kps_vis = data_sample['gt_instances']['keypoints_visible']
                kps_vis[0, absent_indices] = 0
                data_sample['gt_instances']['keypoints_visible'] = kps_vis

            #############
            # 将这些额外信息绑定到 results 列表的最后一个元素上
            # 注意: self.results 是父类维护的列表
            target_result[0]['pred_types'] = pred_type
            target_result[0]['gt_types'] = gt_type
            target_result[0]['gt_instances'] = data_sample['gt_instances']

    def compute_metrics(self, results):
        """
        核心函数：计算 AP，同时手动统计 EPE 均值和 坏点率 (Failure Rate)。
        """
        # 1. 内存清洗 (保持不变)
        if hasattr(self, 'coco') and self.coco is not None:
            for ann_id, ann in self.coco.anns.items():
                types = np.array(ann['keypoint_types'])
                kps = np.array(ann['keypoints']).reshape(-1, 3)
                absent_indices = np.where(types == 2)[0]
                if len(absent_indices) > 0:
                    kps[absent_indices, 2] = 0
                    kps[absent_indices, 0], kps[absent_indices, 1] = 0, 0
                    ann['keypoints'] = kps.flatten().tolist()

        # 2. 调用父类计算标准 AP
        raw_metrics = super().compute_metrics(results)

        # =======================================================
        # 3. 🚀 手动分析：OKS, Avg EPE, 和 Failure Rate
        # =======================================================
        sigmas = self.dataset_meta['sigmas']
        num_kpts = len(sigmas)

        # ⚠️ 定义坏点阈值 (像素)
        # 如果预测偏差超过这个像素值，就算作“坏点”
        # 对于 256x192 或 512x512 的图，通常 30px 已经是很明显的偏差了
        # 你之前的残肢点动不动 400px，那个叫 Horror，这里我们可以设严一点
        FAILURE_THR = 30.0

        kpt_oks_sums = np.zeros(num_kpts)
        kpt_epe_sums = np.zeros(num_kpts)
        kpt_counts = np.zeros(num_kpts)
        kpt_fail_counts = np.zeros(num_kpts)  # ❌ 新增：坏点计数器

        for res in results:
            instance = res[0]
            # 兼容 process 中塞入的数据
            pred_kpts = instance['keypoints'][0]  # [31, 2]

            # 注意：这里要确保你在 process 里存的是 'gt_instances' 还是 'gt_kpts'
            # 假设你按照之前的建议，存的是完整的 gt_instances 字典
            gt_kpt = instance['gt_instances']['keypoints'][0]
            gt_v = instance['gt_instances']['keypoints_visible'][0]

            # 获取 Type (用于排除 Absent 点)
            if 'keypoint_types' in instance['gt_instances']:
                gt_kpt_types = instance['gt_instances']['keypoint_types'][0]
            else:
                gt_kpt_types = np.zeros(num_kpts)

            # 获取 scale
            area = instance.get('area', 100 * 100)
            if isinstance(area, (list, np.ndarray)): area = area[0]
            scale = np.sqrt(area)

            for k in range(num_kpts):
                v_g = gt_v[k]
                kpt_type = gt_kpt_types[k]

                # 如果是 Absent (Type=2)，强制不可见，不参与统计
                if kpt_type == 2:
                    v_g = 0

                if v_g > 0:  # 只统计 GT 存在的点
                    x_g, y_g = gt_kpt[k][:2]
                    x_p, y_p = pred_kpts[k]

                    dist_sq = (x_p - x_g) ** 2 + (y_p - y_g) ** 2
                    dist = np.sqrt(dist_sq)

                    # 计算 OKS
                    oks = np.exp(-dist_sq / (2 * (scale ** 2) * (sigmas[k] ** 2)))

                    # 累加数据
                    kpt_oks_sums[k] += oks
                    kpt_epe_sums[k] += dist
                    kpt_counts[k] += 1

                    # ❌ 判定坏点
                    if dist > FAILURE_THR:
                        kpt_fail_counts[k] += 1

        # 4. 打印报表
        print("\n" + "═" * 85)
        print(f"📉 Failure Threshold: > {FAILURE_THR} pixels")
        print("─" * 85)
        # 增加一列 Fail Rate
        header = f"{'ID':<4} | {'Keypoint Name':<22} | {'Avg OKS':<8} | {'Avg EPE':<8} | {'Fail (>30px)':<15}"
        print(header)
        print("─" * 85)

        for i in range(num_kpts):
            name = self.dataset_meta['keypoint_id2name'].get(i, f"kp_{i}")
            if kpt_counts[i] > 0:
                avg_oks = kpt_oks_sums[i] / kpt_counts[i]
                avg_epe = kpt_epe_sums[i] / kpt_counts[i]

                # 计算失败率
                fail_count = int(kpt_fail_counts[i])
                total_count = int(kpt_counts[i])
                fail_rate = fail_count / total_count

                # 格式化输出: 次数 (百分比)
                fail_str = f"{fail_count}/{total_count} ({fail_rate:.1%})"

                print(f"{i:<4} | {name:<22} | {avg_oks:<8.4f} | {avg_epe:<8.2f} | {fail_str:<15}")
            else:
                print(f"{i:<4} | {name:<22} | {'N/A':<8} | {'N/A':<8} | {'N/A':<15}")

        # 重点看 23-30 号残肢点
        res_idx = [idx for idx in range(23, 31) if kpt_counts[idx] > 0]
        if res_idx:
            res_avg_oks = np.sum(kpt_oks_sums[res_idx]) / np.sum(kpt_counts[res_idx])

            # 统计残肢总坏点率
            res_fail_sum = np.sum(kpt_fail_counts[res_idx])
            res_total_sum = np.sum(kpt_counts[res_idx])
            res_fail_rate = res_fail_sum / res_total_sum

            print("─" * 85)
            print(f"🔥 Residual Limbs (23-30) Summary:")
            print(f"   Total Avg OKS: {res_avg_oks:.4f}")
            print(f"   Failure Rate : {int(res_fail_sum)}/{int(res_total_sum)} ({res_fail_rate:.1%})")
        print("═" * 85 + "\n")

        return raw_metrics