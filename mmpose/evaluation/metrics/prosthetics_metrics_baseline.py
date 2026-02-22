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
                type_probs = data_sample['pred_instances']['type_scores'].cpu().numpy()
            else:
                raise ValueError('Keypoint types are not available in the prediction results.')

            gt_type = data_sample['gt_instances']['keypoint_types'].cpu().numpy()
            types = np.array(gt_type, dtype=np.int64)
            _, absent_indices = np.where(types == 2)

            target_result[0]['pred_types'] = pred_type
            target_result[0]['gt_types'] = gt_type
            target_result[0]['gt_instances'] = data_sample['gt_instances']
            target_result[0]['type_scores'] = type_probs

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

    # =========================================================================
    # 主入口：统计报表调度器
    # =========================================================================
    def report_custom_stats(self, results):
        # 1. 对预测结果应用后处理，并在 instance 中保存 before 和 after 两种标签
        total_corrected = self._apply_post_processing_to_results(results)

        # 2. 分别计算后处理前后的统计指标
        metrics_before = self._calculate_metrics(results, type_key='pred_types_before')
        metrics_after = self._calculate_metrics(results, type_key='pred_types_after')

        # 3. 打印详细对比报表
        self._print_metrics_table(metrics_before, title="🛑 BEFORE Post-Processing (Raw Output)")
        self._print_metrics_table(metrics_after, title="✅ AFTER Post-Processing (Hierarchical Rules)")

        # 4. 打印干预总结与差异
        self._print_intervention_summary(total_corrected, metrics_before, metrics_after)

        # 5. 推送到 Weights & Biases
        self._log_to_wandb(metrics_after, metrics_before, total_corrected)

    # =========================================================================
    # 模块 1：分层启发式后处理 (核心逻辑)
    # =========================================================================
    def _apply_post_processing_to_results(self, results):
        omega_dict = {
            23: [7, 9, 17, 25],
            24: [8, 10, 18, 26],
            25: [9, 17, 23],
            26: [10, 18, 24],
            27: [13, 15, 19, 21, 29],
            28: [14, 16, 20, 22, 30],
            29: [15, 19, 21, 27],
            30: [16, 20, 22, 28],
        }
        limb_residual_pairs = [(23, 25), (24, 26), (27, 29), (28, 30)]

        total_violations_corrected = 0

        for res in results:
            instance = res[0]
            gt_v = instance['gt_instances']['keypoints_visible'][0]
            pred_types_before = instance['pred_types'][0].copy()
            pred_type_scores = instance['type_scores'][0]

            # 保存原始预测
            instance['pred_types_before'] = pred_types_before
            pred_types_after = pred_types_before.copy()

            for upper_r, lower_r in limb_residual_pairs:
                # --- Step 1: 内部肃清 (上下残肢互斥) ---
                if pred_types_after[upper_r] == 0 and pred_types_after[lower_r] == 0:
                    p_up = pred_type_scores[upper_r][0]
                    p_low = pred_type_scores[lower_r][0]
                    if p_up > p_low:
                        pred_types_after[lower_r] = 2
                        if gt_v[lower_r] > 0: total_violations_corrected += 1
                    else:
                        pred_types_after[upper_r] = 2
                        if gt_v[upper_r] > 0: total_violations_corrected += 1

                # --- Step 2: 锁定唯一存活残肢 ---
                active_r = upper_r if pred_types_after[upper_r] == 0 else (
                    lower_r if pred_types_after[lower_r] == 0 else None)

                # --- Step 3: 残肢 vs 下游聚合对抗 ---
                if active_r is not None:
                    downstream_anatomy_nodes = [j for j in omega_dict[active_r] if j < 23]
                    if len(downstream_anatomy_nodes) > 0:
                        avg_down_norm = sum(pred_type_scores[j][0] for j in downstream_anatomy_nodes) / len(
                            downstream_anatomy_nodes)
                        prob_r_normal = pred_type_scores[active_r][0]

                        if prob_r_normal > avg_down_norm:
                            # 残肢胜出，连坐修改下游
                            for j in downstream_anatomy_nodes:
                                if pred_types_after[j] == 0:
                                    p_pros, p_miss = pred_type_scores[j][1], pred_type_scores[j][2]
                                    pred_types_after[j] = 1 if p_pros > p_miss else 2
                                    if gt_v[j] > 0: total_violations_corrected += 1
                        else:
                            # 下游胜出，残肢是幻觉
                            pred_types_after[active_r] = 2
                            if gt_v[active_r] > 0: total_violations_corrected += 1

            instance['pred_types_after'] = pred_types_after

        return total_violations_corrected

    # =========================================================================
    # 模块 2：指标计算引擎 (回归统计全量保留，分类统计仅限四肢残肢)
    # =========================================================================
    def _calculate_metrics(self, results, type_key):
        import numpy as np
        sigmas = self.dataset_meta['sigmas']
        num_kpts = len(sigmas)
        FAILURE_THR = 30.0

        # 🌟 白名单
        LIMB_AND_RES_KPTS = [7, 8, 9, 10, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]

        m = {
            'oks_sums': np.zeros(num_kpts), 'epe_sums': np.zeros(num_kpts), 'counts': np.zeros(num_kpts),
            'fail_counts': np.zeros(num_kpts), 'norm_reg_counts': np.zeros(num_kpts),
            'norm_fail_counts': np.zeros(num_kpts),
            'pros_reg_counts': np.zeros(num_kpts), 'pros_fail_counts': np.zeros(num_kpts),
            'type_counts': np.zeros(num_kpts), 'type_correct': np.zeros(num_kpts),
            'norm_counts': np.zeros(num_kpts), 'norm_correct': np.zeros(num_kpts),
            'pros_counts': np.zeros(num_kpts), 'pros_correct': np.zeros(num_kpts),
            'miss_counts': np.zeros(num_kpts), 'miss_correct': np.zeros(num_kpts),
            'confusion': np.zeros((3, 3), dtype=int),
            'global_correct': 0, 'global_total_valid': 0
        }

        all_y_true, all_y_probs = [], []
        kpt_y_true = [[] for _ in range(num_kpts)]
        kpt_y_probs = [[] for _ in range(num_kpts)]

        for res in results:
            inst = res[0]
            pred_kpts = inst['keypoints'][0]
            gt_kpt = inst['gt_instances']['keypoints'][0]
            gt_v = inst['gt_instances']['keypoints_visible'][0]
            gt_types = inst['gt_instances']['keypoint_types'][0]
            pred_types = inst[type_key]
            type_scores = inst['type_scores'][0]

            area = inst['areas'][0]
            scale = np.sqrt(area[0] if isinstance(area, (list, np.ndarray)) else area)

            for k in range(num_kpts):
                v_g, gt_t, pr_t = gt_v[k], gt_types[k], pred_types[k]

                if k in LIMB_AND_RES_KPTS:
                    if v_g > 0:
                        m['global_total_valid'] += 1
                        if pr_t == gt_t: m['global_correct'] += 1

                        prob_arr = type_scores[k].cpu().numpy() if hasattr(type_scores[k], 'cpu') else type_scores[k]

                        all_y_true.append(gt_t)
                        all_y_probs.append(prob_arr)
                        kpt_y_true[k].append(gt_t)
                        kpt_y_probs[k].append(prob_arr)

                        m['type_counts'][k] += 1
                        if pr_t == gt_t: m['type_correct'][k] += 1
                        if gt_t == 0:
                            m['norm_counts'][k] += 1; m['norm_correct'][k] += (pr_t == 0)
                        elif gt_t == 1:
                            m['pros_counts'][k] += 1; m['pros_correct'][k] += (pr_t == 1)
                        elif gt_t == 2:
                            m['miss_counts'][k] += 1; m['miss_correct'][k] += (pr_t == 2)

                        if k < 23 and 0 <= gt_t <= 2 and 0 <= pr_t <= 2:
                            m['confusion'][gt_t, pr_t] += 1

                # 回归统计
                if gt_t == 2: v_g = 0
                if v_g > 0:
                    dist = np.sqrt((pred_kpts[k][0] - gt_kpt[k][0]) ** 2 + (pred_kpts[k][1] - gt_kpt[k][1]) ** 2)
                    oks = np.exp(-(dist ** 2) / (2 * (scale ** 2) * (sigmas[k] ** 2)))
                    m['oks_sums'][k] += oks
                    m['epe_sums'][k] += dist
                    m['counts'][k] += 1
                    is_fail = dist > FAILURE_THR
                    if is_fail: m['fail_counts'][k] += 1
                    if gt_t == 0:
                        m['norm_reg_counts'][k] += 1; m['norm_fail_counts'][k] += is_fail
                    elif gt_t == 1:
                        m['pros_reg_counts'][k] += 1; m['pros_fail_counts'][k] += is_fail

        # ==========================================
        # 🌟 核心修正：单点 AUC 动态降维计算
        # ==========================================
        m['kpt_auc'] = np.full(num_kpts, np.nan)
        m['macro_auc'] = np.nan
        m['all_y_true'] = np.array(all_y_true)
        m['all_y_probs'] = np.array(all_y_probs)

        try:
            from sklearn.metrics import roc_auc_score
            for i in range(num_kpts):
                y_t = np.array(kpt_y_true[i])
                y_p = np.array(kpt_y_probs[i])

                if len(y_t) > 0:
                    present_classes = np.unique(y_t)
                    if len(present_classes) > 1:
                        # 截取存在的类别列
                        filtered_probs = y_p[:, present_classes]
                        if len(present_classes) == 2:
                            # 二分类需重新归一化
                            filtered_probs = filtered_probs / filtered_probs.sum(axis=1, keepdims=True)
                            m['kpt_auc'][i] = roc_auc_score(y_t, filtered_probs[:, 1])
                        else:
                            # 三分类 OvR
                            m['kpt_auc'][i] = roc_auc_score(y_t, filtered_probs, multi_class='ovr', average='macro')

            if len(m['all_y_true']) > 0 and len(np.unique(m['all_y_true'])) > 1:
                m['macro_auc'] = roc_auc_score(m['all_y_true'], m['all_y_probs'], multi_class='ovr', average='macro')

        except Exception as e:
            import traceback
            print(f"🔥 AUC 计算失败: {e}")
            traceback.print_exc()

        return m

    # =========================================================================
    # 模块 3：打印单一报表表单
    # =========================================================================
    def _print_metrics_table(self, m, title):
        import numpy as np
        num_kpts = len(self.dataset_meta['sigmas'])

        def rate_str(v, t):
            return f"{int(v)}/{int(t)} ({v / t:.1%})" if t > 0 else "N/A"

        # 🌟 统一调整列宽，解决数据量大时的撑爆错位问题
        w_id, w_name, w_oks, w_epe, w_rate, w_auc = 4, 22, 7, 7, 22, 9
        # 动态计算总宽度：所有列宽之和 + 11个分隔符(' | ' = 3个字符)
        total_w = w_id + w_name + w_oks + w_epe + (w_rate * 7) + w_auc + 33

        print("\n" + "═" * total_w)
        print(f"{title}")
        print("─" * total_w)

        header = (f"{'ID':<{w_id}} | {'Keypoint Name':<{w_name}} | {'Avg OKS':<{w_oks}} | {'Avg EPE':<{w_epe}} | "
                  f"{'Fail All':<{w_rate}} | {'Fail Norm':<{w_rate}} | {'Fail Pros':<{w_rate}} | "
                  f"{'All Acc':<{w_rate}} | {'Norm Acc':<{w_rate}} | {'Pros Acc':<{w_rate}} | "
                  f"{'Miss Acc':<{w_rate}} | {'Macro AUC':<{w_auc}}")
        print(header)
        print("─" * total_w)

        for i in range(num_kpts):
            name = self.dataset_meta['keypoint_id2name'].get(i, f"kp_{i}")
            acc_a = rate_str(m['type_correct'][i], m['type_counts'][i])
            acc_n = rate_str(m['norm_correct'][i], m['norm_counts'][i])
            acc_p = rate_str(m['pros_correct'][i], m['pros_counts'][i])
            acc_m = rate_str(m['miss_correct'][i], m['miss_counts'][i])

            if m['counts'][i] > 0:
                o_str = f"{m['oks_sums'][i] / m['counts'][i]:.4f}"
                e_str = f"{m['epe_sums'][i] / m['counts'][i]:.2f}"
                f_a = rate_str(m['fail_counts'][i], m['counts'][i])
                f_n = rate_str(m['norm_fail_counts'][i], m['norm_reg_counts'][i])
                f_p = rate_str(m['pros_fail_counts'][i], m['pros_reg_counts'][i])
            else:
                o_str, e_str, f_a, f_n, f_p = "N/A", "N/A", "N/A", "N/A", "N/A"

            auc_str = f"{m['kpt_auc'][i]:.4f}" if not np.isnan(m['kpt_auc'][i]) else "N/A"

            row = (f"{i:<{w_id}} | {name:<{w_name}} | {o_str:<{w_oks}} | {e_str:<{w_epe}} | "
                   f"{f_a:<{w_rate}} | {f_n:<{w_rate}} | {f_p:<{w_rate}} | "
                   f"{acc_a:<{w_rate}} | {acc_n:<{w_rate}} | {acc_p:<{w_rate}} | "
                   f"{acc_m:<{w_rate}} | {auc_str:<{w_auc}}")
            print(row)

        # 混淆矩阵部分
        print("─" * total_w)
        print("📊 Global Type Confusion Matrix (Excluded Residual Limbs 23-30)")
        tn = ['GT Normal(0)', 'GT Pros(1)', 'GT Miss(2)']
        for i in range(3):
            rt = np.sum(m['confusion'][i])
            s = [f"{m['confusion'][i, j]} ({m['confusion'][i, j] / rt:.1%})" if rt > 0 else "0 (0.0%)" for j in
                 range(3)]
            print(f"   {tn[i]:<14} | {s[0]:<20} | {s[1]:<20} | {s[2]:<20} | {rt}")

        print("═" * total_w + "\n")

    # =========================================================================
    # 模块 4：干预差异总结 (Ablation Comparison)
    # =========================================================================
    def _print_intervention_summary(self, total_corrected, m_before, m_after):
        def r_str(v, t): return f"{v / t:.2%}" if t > 0 else "N/A"

        print("═" * 100)
        print(f"🛡️ Post-Processing Intervention Report & Ablation Study")
        print("─" * 100)
        print(f"   - Anatomical Violations Corrected : {total_corrected} points")
        print(
            f"   - Global Acc BEFORE               : {r_str(m_before['global_correct'], m_before['global_total_valid'])}")
        print(
            f"   - Global Acc AFTER                : {r_str(m_after['global_correct'], m_after['global_total_valid'])}")

        # 计算差异
        diff = m_after['global_correct'] - m_before['global_correct']
        diff_str = f"+{diff}" if diff > 0 else str(diff)
        print(f"   - Net Accuracy Gain               : {diff_str} correct points")

        if not np.isnan(m_after['macro_auc']):
            print(
                f"   - Global Macro AUC (Probabilities): {m_after['macro_auc']:.4f} (Invariant to Post-Processing)")
        print("═" * 100 + "\n")

    # =========================================================================
    # 模块 5：W&B 推送
    # =========================================================================
    def _log_to_wandb(self, m_after, m_before, total_corrected):
        import numpy as np
        try:
            import wandb
            if wandb.run is None:
                return

            all_y_true = m_after['all_y_true']
            all_y_probs = m_after['all_y_probs']

            wandb_metrics = {
                "metrics/macro_auc": m_after['macro_auc'],
                "metrics/violations_corrected": total_corrected,
                "metrics/acc_BEFORE": m_before['global_correct'] / max(1, m_before['global_total_valid']),
                "metrics/acc_AFTER": m_after['global_correct'] / max(1, m_after['global_total_valid']),
            }

            # ==========================================
            # 🌟 强制平衡采样：揭开 20 万 Normal 掩盖的真相
            # ==========================================
            if len(all_y_true) > 0:
                idx_0 = np.where(all_y_true == 0)[0]
                idx_1 = np.where(all_y_true == 1)[0]
                idx_2 = np.where(all_y_true == 2)[0]

                # 为了不被 WandB 随机抹掉少数类，强制每类最多抽取 3333 个点
                # 如果某类不足 3333（极少情况），就全拿
                s_limit = 3333
                s_0 = np.random.choice(idx_0, min(s_limit, len(idx_0)), replace=False) if len(idx_0) > 0 else []
                s_1 = np.random.choice(idx_1, min(s_limit, len(idx_1)), replace=False) if len(idx_1) > 0 else []
                s_2 = np.random.choice(idx_2, min(s_limit, len(idx_2)), replace=False) if len(idx_2) > 0 else []

                balanced_idx = np.concatenate([s_0, s_1, s_2]).astype(int)
                np.random.shuffle(balanced_idx)

                bal_y_true = all_y_true[balanced_idx]
                bal_y_probs = all_y_probs[balanced_idx]

                if len(np.unique(bal_y_true)) > 1:
                    # 记录真实平衡后的图表
                    wandb_metrics["Global_Balanced_ROC"] = wandb.plot.roc_curve(
                        bal_y_true, bal_y_probs, labels=["Normal", "Prosthetic", "Missing"]
                    )
                    wandb_metrics["Global_Balanced_PR"] = wandb.plot.pr_curve(
                        bal_y_true, bal_y_probs, labels=["Normal", "Prosthetic", "Missing"]
                    )

            # 🌟 把每个关节点的 AUC 变成折线图推上去 (干净、清晰、免卡顿)
            for i, auc_val in enumerate(m_after['kpt_auc']):
                if not np.isnan(auc_val):
                    name = self.dataset_meta['keypoint_id2name'].get(i, f"kp_{i}")
                    wandb_metrics[f"AUC_per_kpt/{name}"] = auc_val

            wandb.log(wandb_metrics)

        except Exception as e:
            import traceback
            print(f"🔥 Wandb Log 报错: {e}")
            traceback.print_exc()