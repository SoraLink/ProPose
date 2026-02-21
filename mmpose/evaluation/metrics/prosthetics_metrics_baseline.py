# mmpose/evaluation/metrics/ldpose_metric.py
import numpy as np
from mmpose.registry import METRICS
from mmpose.evaluation.metrics import CocoMetric


@METRICS.register_module()
class ProstheticsMetric(CocoMetric):

    def __init__(self,
                 score_thr=0.3,  # 判定幻觉的置信度阈值
                 **kwargs):
        super().__init__(**kwargs)
        self.score_thr = score_thr
        if hasattr(self, 'coco') and self.coco is not None:
            self._clean_absent_keypoints()

    def _clean_absent_keypoints(self):
        print(f"[{self.__class__.__name__}] Pre-processing COCO annotations for Absent joints...")

        for ann_id, ann in self.coco.anns.items():
            if 'keypoint_types' not in ann:
                continue

            types = np.array(ann['keypoint_types'])
            absent_indices = np.where(types == 2)[0]

            if len(absent_indices) > 0:
                kps = np.array(ann['keypoints']).reshape(-1, 3)

                kps[absent_indices, 2] = 0
                kps[absent_indices, 0] = 0
                kps[absent_indices, 1] = 0

                ann['keypoints'] = kps.flatten().tolist()

        print(f"[{self.__class__.__name__}] Pre-processing complete.")

    def process(self, data_batch, data_samples):

        super().process(data_batch, data_samples)
        batch_len = len(data_samples)
        start_idx = len(self.results) - batch_len

        for i, data_sample in enumerate(data_samples):

            target_result = self.results[start_idx + i]

            if 'keypoint_types' in data_sample['pred_instances']:
                pred_type = data_sample['pred_instances']['keypoint_types'].cpu().numpy()
            else:
                num_kps = data_sample['pred_instances']['keypoints'].shape[1]
                pred_type = np.zeros((num_kps,), dtype=int)

            gt_type = data_sample['gt_instances']['keypoint_types'].cpu().numpy()
            types = np.array(gt_type, dtype=np.int64)
            _, absent_indices = np.where(types == 2)

            target_result[0]['pred_types'] = pred_type
            target_result[0]['gt_types'] = gt_type
            target_result[0]['gt_instances'] = data_sample['gt_instances']

    def compute_metrics(self, results):

        raw_metrics = super().compute_metrics(results)
        total_kpts = 0
        correct_types = 0

        res_total = 0
        res_correct = 0

        for res in results:
            instance = res[0]
            p_types = np.array(instance['pred_types']).flatten()
            g_types = np.array(instance['gt_types']).flatten()
            vis = np.array(instance['gt_instances']['keypoints_visible']).flatten()

            valid_mask = vis > 0

            valid_p_types = p_types[valid_mask]
            valid_g_types = g_types[valid_mask]

            correct_types += np.sum(valid_p_types == valid_g_types)
            total_kpts += len(valid_g_types)

            res_mask = valid_mask[23:31]
            res_pred = p_types[23:31][res_mask]
            res_gt = g_types[23:31][res_mask]

            res_correct += np.sum(res_pred == res_gt)
            res_total += len(res_gt)

        if total_kpts > 0:
            raw_metrics['Type_Acc_All'] = correct_types / total_kpts

        if res_total > 0:
            raw_metrics['Type_Acc_Residual'] = res_correct / res_total

        self.report_custom_stats(results)

        return raw_metrics

    def report_custom_stats(self, results):
        sigmas = self.dataset_meta['sigmas']
        num_kpts = len(sigmas)

        FAILURE_THR = 30.0

        # 回归误差统计 (总体)
        kpt_oks_sums = np.zeros(num_kpts)
        kpt_epe_sums = np.zeros(num_kpts)
        kpt_counts = np.zeros(num_kpts)
        kpt_fail_counts = np.zeros(num_kpts)

        # 回归误差统计 (细分 Type 0 和 Type 1)
        kpt_norm_reg_counts = np.zeros(num_kpts)
        kpt_norm_fail_counts = np.zeros(num_kpts)
        kpt_pros_reg_counts = np.zeros(num_kpts)
        kpt_pros_fail_counts = np.zeros(num_kpts)

        # 分类准确率统计
        kpt_type_counts = np.zeros(num_kpts)
        kpt_type_correct_counts = np.zeros(num_kpts)
        kpt_missing_type_counts = np.zeros(num_kpts)
        kpt_missing_type_correct_counts = np.zeros(num_kpts)
        kpt_pros_type_counts = np.zeros(num_kpts)
        kpt_pros_type_correct_counts = np.zeros(num_kpts)
        kpt_normal_type_counts = np.zeros(num_kpts)
        kpt_normal_type_correct_counts = np.zeros(num_kpts)

        # 用于统计 GT Type 到 Pred Type 的混淆矩阵 (3x3)
        type_confusion_matrix = np.zeros((3, 3), dtype=int)

        for res in results:
            instance = res[0]
            pred_kpts = instance['keypoints'][0]  # [31, 2]

            gt_kpt = instance['gt_instances']['keypoints'][0]
            gt_v = instance['gt_instances']['keypoints_visible'][0]

            if 'keypoint_types' in instance['gt_instances']:
                gt_kpt_types = instance['gt_instances']['keypoint_types'][0]
                pred_kpt_types = instance['pred_types'][0]
            else:
                raise ValueError("Keypoint types are not available.")

            # 获取 scale
            area = instance['areas'][0]
            if isinstance(area, (list, np.ndarray)): area = area[0]
            scale = np.sqrt(area)

            for k in range(num_kpts):
                v_g = gt_v[k]
                kpt_type = gt_kpt_types[k]
                pred_type = pred_kpt_types[k]

                # --- 1. Type 分类准确率统计 ---
                if v_g > 0:
                    kpt_type_counts[k] += 1
                    if pred_type == kpt_type:
                        kpt_type_correct_counts[k] += 1

                    if kpt_type == 0:
                        kpt_normal_type_counts[k] += 1
                        if pred_type == 0:
                            kpt_normal_type_correct_counts[k] += 1

                    if kpt_type == 1:
                        kpt_pros_type_counts[k] += 1
                        if pred_type == 1:
                            kpt_pros_type_correct_counts[k] += 1

                    if kpt_type == 2:
                        kpt_missing_type_counts[k] += 1
                        if pred_type == 2:
                            kpt_missing_type_correct_counts[k] += 1

                    # 记录混淆矩阵数据 (仅对 0-22 基础关键点)
                    if k < 23:
                        if 0 <= kpt_type <= 2 and 0 <= pred_type <= 2:
                            type_confusion_matrix[kpt_type, pred_type] += 1

                # --- 2. 回归距离与失败率统计 ---
                # 如果是 Absent (Type=2)，强制不可见，不参与回归统计
                if kpt_type == 2:
                    v_g = 0

                if v_g > 0:  # 只统计 GT 存在的点 (Type 0 和 Type 1)
                    x_g, y_g = gt_kpt[k][:2]
                    x_p, y_p = pred_kpts[k]

                    dist_sq = (x_p - x_g) ** 2 + (y_p - y_g) ** 2
                    dist = np.sqrt(dist_sq)

                    # 计算 OKS
                    oks = np.exp(-dist_sq / (2 * (scale ** 2) * (sigmas[k] ** 2)))

                    # 累加全局数据
                    kpt_oks_sums[k] += oks
                    kpt_epe_sums[k] += dist
                    kpt_counts[k] += 1

                    is_fail = dist > FAILURE_THR
                    if is_fail:
                        kpt_fail_counts[k] += 1

                    # 细分回归数据到 Normal 和 Prosthetic
                    if kpt_type == 0:
                        kpt_norm_reg_counts[k] += 1
                        if is_fail:
                            kpt_norm_fail_counts[k] += 1
                    elif kpt_type == 1:
                        kpt_pros_reg_counts[k] += 1
                        if is_fail:
                            kpt_pros_fail_counts[k] += 1

        # 格式化输出函数
        def get_rate_str(val, total):
            return f"{int(val)}/{int(total)} ({val / total:.1%})" if total > 0 else "N/A"

        # 表格加宽到 185 以容纳新增列
        print("\n" + "═" * 185)
        print(f"📉 Failure Threshold: > {FAILURE_THR} pixels")
        print("─" * 185)

        header = f"{'ID':<4} | {'Keypoint Name':<22} | {'Avg OKS':<7} | {'Avg EPE':<7} | {'Fail All':<14} | {'Fail Norm':<14} | {'Fail Pros':<14} | {'All Acc':<18} | {'Norm Acc':<18} | {'Pros Acc':<18} | {'Miss Acc':<18}"
        print(header)
        print("─" * 185)

        for i in range(num_kpts):
            name = self.dataset_meta['keypoint_id2name'].get(i, f"kp_{i}")

            # Type 分类 Accurate 字符串
            acc_all = get_rate_str(kpt_type_correct_counts[i], kpt_type_counts[i])
            acc_norm = get_rate_str(kpt_normal_type_correct_counts[i], kpt_normal_type_counts[i])
            acc_pros = get_rate_str(kpt_pros_type_correct_counts[i], kpt_pros_type_counts[i])
            acc_miss = get_rate_str(kpt_missing_type_correct_counts[i], kpt_missing_type_counts[i])

            # 回归指标字符串
            if kpt_counts[i] > 0:
                avg_oks = kpt_oks_sums[i] / kpt_counts[i]
                avg_epe = kpt_epe_sums[i] / kpt_counts[i]

                oks_str = f"{avg_oks:.4f}"
                epe_str = f"{avg_epe:.2f}"

                fail_all_str = get_rate_str(kpt_fail_counts[i], kpt_counts[i])
                fail_norm_str = get_rate_str(kpt_norm_fail_counts[i], kpt_norm_reg_counts[i])
                fail_pros_str = get_rate_str(kpt_pros_fail_counts[i], kpt_pros_reg_counts[i])
            else:
                oks_str, epe_str = "N/A", "N/A"
                fail_all_str, fail_norm_str, fail_pros_str = "N/A", "N/A", "N/A"

            # 打印单行
            print(
                f"{i:<4} | {name:<22} | {oks_str:<7} | {epe_str:<7} | {fail_all_str:<14} | {fail_norm_str:<14} | {fail_pros_str:<14} | {acc_all:<18} | {acc_norm:<18} | {acc_pros:<18} | {acc_miss:<18}")

        # ---------------------------------------------------------
        # 全局 Type 混淆矩阵报表 (Excluded Residual Limbs 23-30)
        # ---------------------------------------------------------
        print("─" * 185)
        print("📊 Global Type Confusion Matrix (Excluded Residual Limbs 23-30, Row: GT, Col: Pred)")
        print(f"   {'':<14} | {'Pred Normal (0)':<20} | {'Pred Pros (1)':<20} | {'Pred Miss (2)':<20} | {'Total GT'}")

        type_names = ['GT Normal (0)', 'GT Pros (1)', 'GT Miss (2)']
        for i in range(3):
            row_total = np.sum(type_confusion_matrix[i])
            if row_total > 0:
                p0 = type_confusion_matrix[i, 0] / row_total
                p1 = type_confusion_matrix[i, 1] / row_total
                p2 = type_confusion_matrix[i, 2] / row_total
                s0 = f"{type_confusion_matrix[i, 0]} ({p0:.1%})"
                s1 = f"{type_confusion_matrix[i, 1]} ({p1:.1%})"
                s2 = f"{type_confusion_matrix[i, 2]} ({p2:.1%})"
            else:
                s0, s1, s2 = "0 (0.0%)", "0 (0.0%)", "0 (0.0%)"

            print(f"   {type_names[i]:<14} | {s0:<20} | {s1:<20} | {s2:<20} | {row_total}")

        # ---------------------------------------------------------
        # 汇总残肢数据 (Residual Limbs 23-30)
        # ---------------------------------------------------------
        res_idx = [idx for idx in range(23, 31) if kpt_counts[idx] > 0]
        res_type_idx = [idx for idx in range(23, 31) if kpt_type_counts[idx] > 0]

        if res_idx or res_type_idx:
            print("─" * 185)
            print(f"🔥 Residual Limbs (23-30) Summary:")

            if res_idx:
                res_avg_oks = np.sum(kpt_oks_sums[res_idx]) / np.sum(kpt_counts[res_idx])

                res_fail_sum = np.sum(kpt_fail_counts[res_idx])
                res_total_sum = np.sum(kpt_counts[res_idx])
                res_fail_norm_sum = np.sum(kpt_norm_fail_counts[res_idx])
                res_norm_tot = np.sum(kpt_norm_reg_counts[res_idx])
                res_fail_pros_sum = np.sum(kpt_pros_fail_counts[res_idx])
                res_pros_tot = np.sum(kpt_pros_reg_counts[res_idx])

                print(f"   Total Avg OKS: {res_avg_oks:.4f}")
                print(f"   Failure Rate : {get_rate_str(res_fail_sum, res_total_sum)}")
                print(f"     - Fail Normal  : {get_rate_str(res_fail_norm_sum, res_norm_tot)}")
                print(f"     - Fail Pros    : {get_rate_str(res_fail_pros_sum, res_pros_tot)}")

            if res_type_idx:
                tot_correct = np.sum(kpt_type_correct_counts[res_type_idx])
                tot_count = np.sum(kpt_type_counts[res_type_idx])
                norm_correct = np.sum(kpt_normal_type_correct_counts[res_type_idx])
                norm_count = np.sum(kpt_normal_type_counts[res_type_idx])
                pros_correct = np.sum(kpt_pros_type_correct_counts[res_type_idx])
                pros_count = np.sum(kpt_pros_type_counts[res_type_idx])
                miss_correct = np.sum(kpt_missing_type_correct_counts[res_type_idx])
                miss_count = np.sum(kpt_missing_type_counts[res_type_idx])

                print(f"   Overall Type Acc : {get_rate_str(tot_correct, tot_count)}")
                print(f"     - Normal Acc   : {get_rate_str(norm_correct, norm_count)}")
                print(f"     - Pros Acc     : {get_rate_str(pros_correct, pros_count)}")
                print(f"     - Missing Acc  : {get_rate_str(miss_correct, miss_count)}")

        print("═" * 185 + "\n")