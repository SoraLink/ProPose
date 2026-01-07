# mmpose/evaluation/metrics/ldpose_metric.py
import copy
import numpy as np
from mmpose.registry import METRICS
from mmpose.evaluation.metrics import CocoMetric


@METRICS.register_module()
class ProstheticsMetric(CocoMetric):
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
                 score_thr=0.3,  # 判定幻觉的置信度阈值
                 **kwargs):
        super().__init__(**kwargs)
        self.score_thr = score_thr

        # 定义解剖链: Root -> Children
        # 逻辑: 如果 Root 是残肢(GT存在)，Children 应该全黑。如果 Children 亮了，罚 Root。
        self.chain_dependency = {
            # 上肢
            17: [7, 9],  # 左上臂残肢 -> 肘, 腕
            18: [8, 10],  # 右上臂残肢 -> 肘, 腕
            19: [9],  # 左前臂残肢 -> 腕
            20: [10],  # 右前臂残肢 -> 腕
            # 下肢
            21: [13, 15],  # 左大腿残肢 -> 膝, 踝
            22: [14, 16],  # 右大腿残肢 -> 膝, 踝
            23: [15],  # 左小腿残肢 -> 踝
            24: [16]  # 右小腿残肢 -> 踝
        }

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
                num_kps = data_sample['pred_instances'].keypoints.shape[1]
                # 如果没有 Type Head，默认全 0 (Normal)
                pred_type = np.zeros((num_kps,), dtype=int)

            # 存 GT Type
            gt_type = data_sample['gt_instances']['keypoint_types'].cpu().numpy()

            # 将这些额外信息绑定到 results 列表的最后一个元素上
            # 注意: self.results 是父类维护的列表
            target_result[0]['pred_types'] = pred_type
            target_result[0]['gt_types'] = gt_type

    def compute_metrics(self, results):
        """
        核心函数：在计算 AP 之前，先执行惩罚逻辑 (The Purge)。
        """

        raw_metrics = super().compute_metrics(results)
        final_metrics = {}
        for k, v in raw_metrics.items():
            final_metrics[f'Standard_{k}'] = v  # e.g., Standard_coco/AP
        # 1. 深拷贝一份结果，避免污染原始数据
        eval_results = copy.deepcopy(results)

        ghost_cnt = 0
        total_missing_cnt = 0

        type_correct_all = 0
        type_total_all = 0

        type_correct_vis = 0
        type_total_vis = 0

        # 2. 遍历所有样本进行"惩罚"
        for i, res in enumerate(eval_results):
            # 获取该样本的预测和 GT
            # 注意：CocoMetric 的 results 格式通常包含 'keypoints' (Kx2) and 'keypoint_scores' (K)
            pred_scores = res[0]['keypoint_scores'][0]

            # 我们刚才存在里面的 Types
            pred_types = res[0].get('pred_types', np.zeros(len(pred_scores)))[0]
            gt_types = res[0].get('gt_types', np.zeros(len(pred_scores)))[0]

            match_mask = (pred_types == gt_types)
            type_correct_all += np.sum(match_mask)
            type_total_all += len(gt_types)
            # 获取 GT 的可见性 (v) 用于判断 Missing
            # 在 eval 阶段通常需要从 self.dataset 或原始标注获取 GT 的具体信息
            # 这里简化处理：我们假设 gt_types=2 就是 Missing

            vis_mask = (gt_types != 2)
            vis_count = np.sum(vis_mask)

            if vis_count > 0:
                type_total_vis += vis_count
                # 在可见点的掩码下，看预测是否正确
                vis_correct = match_mask[vis_mask]
                type_correct_vis += np.sum(vis_correct)

            for k in range(len(pred_scores)):
                # 如果 GT 标记为 Missing (2)
                if gt_types[k] == 2:
                    total_missing_cnt += 1  # 分母：总缺失点数

                    # 判定幻觉条件：置信度高 且 预测类型认为存在
                    is_high_conf = pred_scores[k] > self.score_thr
                    is_pred_exist = pred_types[k] != 2

                    if is_high_conf and is_pred_exist:
                        ghost_cnt += 1

            # === A. 材质感知惩罚 (Type-Aware Penalty) ===
            # 策略：如果 GT 存在 (v>0) 且类型不对，直接把 Score 清零
            for k in range(len(pred_scores)):
                if gt_types[k] != 2 and pred_types[k] != gt_types[k]:
                    pred_scores[k] = 0.0

            # === B. 连坐惩罚 (Chain Penalty) ===
            # 策略：如果 Root 是残肢，但预测出了 Child，罚 Root
            for root_idx, child_indices in self.chain_dependency.items():
                root_idx = int(root_idx)
                if gt_types[root_idx] != 2:
                    has_hallucination = False
                    for child_idx in child_indices:
                        if gt_types[child_idx] == 2:
                            is_high_conf = pred_scores[child_idx] > self.score_thr
                            is_pred_exist = pred_types[child_idx] != 2
                            if is_high_conf and is_pred_exist:
                                has_hallucination = True

                    # 如果发现下游有幻觉，连坐惩罚 Root
                    if has_hallucination:
                        pred_scores[root_idx] = 0.0  # Root 连坐处死！

            # 更新分数回结果列表
            res[0]['keypoint_scores'] = pred_scores[None]

        # 3. 计算 Ghost Rate
        ghost_rate = 0.0
        if total_missing_cnt > 0:
            ghost_rate = ghost_cnt / total_missing_cnt

        acc_all = type_correct_all / type_total_all if type_total_all > 0 else 0.0
        acc_vis = type_correct_vis / type_total_vis if type_total_vis > 0 else 0.0

        # 4. 调用父类方法计算标准 AP (此时分数已经被我们惩罚过了)
        # 这里的 results 已经被修改了 scores
        ld_metrics = super().compute_metrics(eval_results)
        # 5. 把 Ghost Rate 加进输出字典
        for k, v in ld_metrics.items():
            final_metrics[f'LDPros_{k}'] = v

        final_metrics['Ghost_Rate'] = ghost_rate
        final_metrics['Type_Acc_All'] = acc_all
        final_metrics['Type_Acc_Vis'] = acc_vis

        return final_metrics