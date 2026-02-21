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

        kpt_oks_sums = np.zeros(num_kpts)
        kpt_epe_sums = np.zeros(num_kpts)
        kpt_counts = np.zeros(num_kpts)
        kpt_fail_counts = np.zeros(num_kpts)
        kpt_type_counts = np.zeros(num_kpts)
        kpt_type_correct_counts = np.zeros(num_kpts)
        kpt_missing_type_counts = np.zeros(num_kpts)
        kpt_missing_type_correct_counts = np.zeros(num_kpts)
        kpt_pros_type_counts = np.zeros(num_kpts)
        kpt_pros_type_correct_counts = np.zeros(num_kpts)
        kpt_normal_type_counts = np.zeros(num_kpts)
        kpt_normal_type_correct_counts = np.zeros(num_kpts)

        # 新增：用于统计 GT Type 到 Pred Type 的混淆矩阵 (3x3)
        # 行代表 GT Type (0, 1, 2)，列代表 Pred Type (0, 1, 2)
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

                    # 新增：记录混淆矩阵数据
                    if k < 23:
                        if 0 <= kpt_type <= 2 and 0 <= pred_type <= 2:
                            type_confusion_matrix[kpt_type, pred_type] += 1

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

        # 更新：将返回值改成包含 数量 和 比例 的字符串
        def get_acc_str(correct, total):
            return f"{int(correct)}/{int(total)} ({correct / total:.1%})" if total > 0 else "N/A"

        # 表格加宽到 155
        print("\n" + "═" * 155)
        print(f"📉 Failure Threshold: > {FAILURE_THR} pixels")
        print("─" * 155)

        # 表头列宽相应调大
        header = f"{'ID':<4} | {'Keypoint Name':<22} | {'Avg OKS':<7} | {'Avg EPE':<7} | {'Fail (>30px)':<14} | {'All Acc':<18} | {'Norm Acc':<18} | {'Pros Acc':<18} | {'Miss Acc':<18}"
        print(header)
        print("─" * 155)

        for i in range(num_kpts):
            name = self.dataset_meta['keypoint_id2name'].get(i, f"kp_{i}")

            # 计算细分 Type Accuracy
            acc_all = get_acc_str(kpt_type_correct_counts[i], kpt_type_counts[i])
            acc_norm = get_acc_str(kpt_normal_type_correct_counts[i], kpt_normal_type_counts[i])
            acc_pros = get_acc_str(kpt_pros_type_correct_counts[i], kpt_pros_type_counts[i])
            acc_miss = get_acc_str(kpt_missing_type_correct_counts[i], kpt_missing_type_counts[i])

            # 计算回归指标
            if kpt_counts[i] > 0:
                avg_oks = kpt_oks_sums[i] / kpt_counts[i]
                avg_epe = kpt_epe_sums[i] / kpt_counts[i]
                fail_count = int(kpt_fail_counts[i])
                total_count = int(kpt_counts[i])
                fail_rate = fail_count / total_count

                oks_str = f"{avg_oks:.4f}"
                epe_str = f"{avg_epe:.2f}"
                fail_str = f"{fail_count}/{total_count} ({fail_rate:.1%})"
            else:
                oks_str, epe_str, fail_str = "N/A", "N/A", "N/A"

            # 打印单行
            print(
                f"{i:<4} | {name:<22} | {oks_str:<7} | {epe_str:<7} | {fail_str:<14} | {acc_all:<18} | {acc_norm:<18} | {acc_pros:<18} | {acc_miss:<18}")

        # ---------------------------------------------------------
        # 新增：全局 Type 混淆矩阵报表 (Global Type Confusion Matrix)
        # ---------------------------------------------------------
        print("─" * 155)
        print("📊 Global Type Confusion Matrix (Row: GT Type, Col: Pred Type)")
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
            print("─" * 155)
            print(f"🔥 Residual Limbs (23-30) Summary:")

            if res_idx:
                res_avg_oks = np.sum(kpt_oks_sums[res_idx]) / np.sum(kpt_counts[res_idx])
                res_fail_sum = np.sum(kpt_fail_counts[res_idx])
                res_total_sum = np.sum(kpt_counts[res_idx])
                res_fail_rate = res_fail_sum / res_total_sum

                print(f"   Total Avg OKS: {res_avg_oks:.4f}")
                print(f"   Failure Rate : {int(res_fail_sum)}/{int(res_total_sum)} ({res_fail_rate:.1%})")

            if res_type_idx:
                tot_correct = np.sum(kpt_type_correct_counts[res_type_idx])
                tot_count = np.sum(kpt_type_counts[res_type_idx])

                norm_correct = np.sum(kpt_normal_type_correct_counts[res_type_idx])
                norm_count = np.sum(kpt_normal_type_counts[res_type_idx])

                pros_correct = np.sum(kpt_pros_type_correct_counts[res_type_idx])
                pros_count = np.sum(kpt_pros_type_counts[res_type_idx])

                miss_correct = np.sum(kpt_missing_type_correct_counts[res_type_idx])
                miss_count = np.sum(kpt_missing_type_counts[res_type_idx])

                # 这里的 get_acc_str 已经包含了 count，所以直接调用即可，不需要额外再拼字符串了
                print(f"   Overall Type Acc : {get_acc_str(tot_correct, tot_count)}")
                print(f"     - Normal Acc   : {get_acc_str(norm_correct, norm_count)}")
                print(f"     - Pros Acc     : {get_acc_str(pros_correct, pros_count)}")
                print(f"     - Missing Acc  : {get_acc_str(miss_correct, miss_count)}")

        print("═" * 155 + "\n")