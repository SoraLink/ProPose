import cv2
import mmcv
import os
import copy
import numpy as np
import torch
import mmengine
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmpose.registry import DATASETS, VISUALIZERS
from mmpose.structures import PoseDataSample
from mmengine.structures import InstanceData
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
import numpy as np


# ==========================================
# 1. 配置区域 (请修改这里)
# ==========================================
CONFIG_FILE = 'configs/body_2d_keypoint/topdown_heatmap/coco/VIT_prosthetics.py'
CHECKPOINT_FILE = 'work_dirs/VIT_prosthetics/epoch_150.pth'  # 仅用于初始化 visualizer
RESULT_FILE = 'base_VIT_results.pkl'  # test.py 跑出来的结果
OUT_DIR = './bad_case_analysis'
SCORE_THR = -100.0  # 幻觉判定阈值，与你 Metric 中一致


# ==========================================
# 2. 辅助工具
# ==========================================
def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, list):
        return np.array(x)
    return np.array(x)


def native_compute_oks(pred, gt, area, sigmas, gt_vis_mask=None):
    if gt_vis_mask is None: gt_vis_mask = np.ones(len(gt), dtype=bool)
    pred = to_numpy(pred);
    gt = to_numpy(gt);
    sigmas = to_numpy(sigmas)
    pred_v = pred[gt_vis_mask];
    gt_v = gt[gt_vis_mask];
    sigmas_v = sigmas[gt_vis_mask]
    if len(pred_v) == 0: return 0.0
    dx = pred_v[:, 0] - gt_v[:, 0];
    dy = pred_v[:, 1] - gt_v[:, 1]
    e = (dx ** 2 + dy ** 2) / (2 * area * (sigmas_v ** 2) + 1e-9)
    return np.mean(np.exp(-e))


# ==========================================
# 3. 物理布局引擎 (核心算法)
# ==========================================
def optimize_label_positions(kps, scores, center, score_thr, sidebar_width, img_w, img_h):
    """
    使用简易力导向算法优化标签位置，防止重叠。
    """
    # 筛选出需要绘制的点
    valid_indices = [i for i, s in enumerate(scores) if s >= score_thr]
    if not valid_indices:
        return {}

    # 初始化位置：默认向四周发散
    # 初始引线长度设为 30
    init_positions = []
    anchors = []  # 对应的关键点坐标(钉子)

    for i in valid_indices:
        kp = kps[i]
        anchors.append(kp)

        # 初始方向：从中心指向关键点
        vec = kp - center
        norm = np.linalg.norm(vec)
        direction = vec / norm if norm > 1e-3 else np.array([0, -1])

        # 初始位置
        pos = kp + direction * 30
        init_positions.append(pos)

    positions = np.array(init_positions)
    anchors = np.array(anchors)

    # === 迭代优化 (Simulated Annealing 简化版) ===
    # 迭代参数
    ITERATIONS = 50
    REPULSION_DIST = 25  # 标签之间的最小距离 (斥力半径)
    MAX_LEADER_LEN = 80  # 最大引线长度 (拉力限制)
    MIN_LEADER_LEN = 20  # 最小引线长度
    REPULSION_FORCE = 0.5
    ATTRACTION_FORCE = 0.1

    for _ in range(ITERATIONS):
        # 1. 计算斥力 (Label vs Label)
        # 对每一对标签，如果距离太近，就互相推开
        num_pts = len(positions)
        movements = np.zeros_like(positions)

        for i in range(num_pts):
            for j in range(i + 1, num_pts):
                diff = positions[i] - positions[j]
                dist = np.linalg.norm(diff)

                if dist < REPULSION_DIST:
                    # 距离太近，产生斥力
                    if dist < 1e-3:  # 防止重合除以0
                        force_dir = np.random.randn(2)
                        force_dir /= np.linalg.norm(force_dir)
                    else:
                        force_dir = diff / dist

                    # 距离越近，斥力越大
                    force_mag = (REPULSION_DIST - dist) * REPULSION_FORCE
                    movements[i] += force_dir * force_mag
                    movements[j] -= force_dir * force_mag

        # 2. 计算弹力 (Label vs Anchor Keypoint)
        # 标签不能离关键点太远，也不能太近
        for i in range(num_pts):
            diff = positions[i] - anchors[i]
            dist = np.linalg.norm(diff)

            # 如果太远，拉回来
            if dist > MAX_LEADER_LEN:
                force_dir = -(diff / dist)
                movements[i] += force_dir * (dist - MAX_LEADER_LEN) * ATTRACTION_FORCE
            # 如果太近，推出去
            elif dist < MIN_LEADER_LEN:
                force_dir = (diff / dist) if dist > 1e-3 else np.array([0, 1])
                movements[i] += force_dir * (MIN_LEADER_LEN - dist) * ATTRACTION_FORCE

        # 应用移动
        positions += movements

        # 3. 边界限制 (Clamp)
        # 必须在右侧画布内，不能进入左侧 Sidebar，不能出上下界
        # 注意：这里的坐标是相对于原图的，绘制时会统一加上 SIDEBAR_WIDTH
        # 所以 x 必须 >= 0 (对应实际 SIDEBAR_WIDTH), <= img_w
        positions[:, 0] = np.clip(positions[:, 0], 5, img_w - 5)
        positions[:, 1] = np.clip(positions[:, 1], 10, img_h - 10)

    # 返回优化后的位置字典 {index: (x, y)}
    optimized_map = {}
    for idx_in_list, real_idx in enumerate(valid_indices):
        optimized_map[real_idx] = positions[idx_in_list]

    return optimized_map


# ==========================================
# 4. Analyzer (保持不变)
# ==========================================
class LDPoseAnalyzer:
    def __init__(self, dataset_sigmas=None):
        self.chain_dependency = {
            17: [7, 9], 18: [8, 10], 19: [9], 20: [10],
            21: [13, 15], 22: [14, 16], 23: [15], 24: [16]
        }
        self.score_thr = SCORE_THR
        self.sigmas = dataset_sigmas if dataset_sigmas is not None else \
            np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89]) / 10.0

    def analyze_instance(self, pred_inst, gt_inst, gt_area):
        pred_kps = to_numpy(pred_inst['keypoints'])
        pred_scores = to_numpy(pred_inst['keypoint_scores'])
        raw_pred_types = pred_inst.get('keypoint_types', None)
        pred_types = to_numpy(raw_pred_types).astype(int) if raw_pred_types is not None else np.zeros(len(pred_scores),
                                                                                                      dtype=int)

        gt_kps = to_numpy(gt_inst['keypoints'])
        gt_types = to_numpy(gt_inst['keypoint_types']).astype(int)
        if isinstance(gt_area, (np.ndarray, torch.Tensor)): gt_area = float(gt_area)

        valid_mask = (gt_types != 2)

        ghost_points = []
        for k in range(len(pred_scores)):
            if gt_types[k] == 2:
                if pred_scores[k] > self.score_thr and pred_types[k] != 2:
                    ghost_points.append(k)

        match_count = np.sum(pred_types == gt_types)
        total_kps = len(gt_types)
        type_acc = match_count / total_kps if total_kps > 0 else 0.0

        std_oks = native_compute_oks(pred_kps, gt_kps, gt_area, self.sigmas, valid_mask)

        penalized_kps = pred_kps.copy()
        for k in range(len(pred_scores)):
            if gt_types[k] != 2 and pred_types[k] != gt_types[k]:
                penalized_kps[k] = [-10000, -10000]
        for root_idx, child_indices in self.chain_dependency.items():
            if gt_types[root_idx] != 2:
                has_hallucination = False
                for child_idx in child_indices:
                    if gt_types[child_idx] == 2:
                        if pred_scores[child_idx] > self.score_thr and pred_types[child_idx] != 2:
                            has_hallucination = True;
                            break
                if has_hallucination: penalized_kps[root_idx] = [-10000, -10000]

        ld_oks = native_compute_oks(penalized_kps, gt_kps, gt_area, self.sigmas, valid_mask)

        return {
            'ghost_cnt': len(ghost_points),
            'std_oks': std_oks,
            'ld_oks': ld_oks,
            'type_acc': type_acc,
            'ghost_indices': ghost_points
        }


# ==========================================
# 5. 增强版绘图函数 (已修改 Type Error 统计逻辑)
# ==========================================
def draw_enhanced(candidates, folder, metric_fmt, visualizer, score_thr=SCORE_THR):
    TYPE_MAP = {0: 'N', 1: 'P', 2: 'M'}
    TYPE_COLOR_MAP = {
        0: (0, 255, 0),  # Green
        1: (255, 255, 0),  # Cyan
        2: (128, 128, 128)  # Gray
    }
    SIDEBAR_WIDTH = 350

    for rank, item in enumerate(candidates):
        img_path = item['img_path']
        img_bgr = mmcv.imread(img_path)
        img_rgb = mmcv.imconvert(img_bgr, 'bgr', 'rgb')
        h, w, c = img_bgr.shape

        # 1. 绘制基础骨架
        data_sample = PoseDataSample()
        data_sample.set_metainfo(item['gt_info'])
        pred_inst = InstanceData()
        pred_inst.keypoints = item['pred_all_numpy']['keypoints']
        pred_inst.keypoint_scores = item['pred_all_numpy']['keypoint_scores']
        data_sample.pred_instances = pred_inst

        visualizer.add_datasample(
            os.path.basename(img_path),
            img_rgb,
            data_sample=data_sample,
            draw_gt=False,
            draw_bbox=True,
            show=False,
            out_file=None,
            step=rank
        )
        skeleton_img_rgb = visualizer.get_image()
        skeleton_img_bgr = mmcv.imconvert(skeleton_img_rgb, 'rgb', 'bgr')

        # 2. 创建侧边栏画布
        canvas_bgr = cv2.copyMakeBorder(
            skeleton_img_bgr, 0, 0, SIDEBAR_WIDTH, 0,
            cv2.BORDER_CONSTANT, value=(255, 255, 255)
        )

        # 3. 准备数据
        target_inst_id = item['inst_id']
        kps = item['pred_all_numpy']['keypoints'][target_inst_id]
        scores = item['pred_all_numpy']['keypoint_scores'][target_inst_id]
        if 'keypoint_types' in item['pred_all_numpy']:
            pred_types = item['pred_all_numpy']['keypoint_types'][target_inst_id]
        else:
            pred_types = np.zeros(len(scores), dtype=int)
        gt_types = item['gt_types_numpy']

        # 计算中心
        valid_mask = scores > score_thr
        if np.sum(valid_mask) > 0:
            inst_center = np.mean(kps[valid_mask], axis=0)
        else:
            inst_center = np.array([w / 2, h / 2])

        # === 4. 运行物理引擎优化标签位置 ===
        optimized_positions = optimize_label_positions(
            kps, scores, inst_center, score_thr, SIDEBAR_WIDTH, w, h
        )

        # === 5. 绘制文字和引线 (图层分离策略) ===
        draw_queue = []

        for i, (kp, score, p_type) in enumerate(zip(kps, scores, pred_types)):
            if score >= score_thr and i in optimized_positions:
                p_type_int = int(p_type)
                label_text = f"{i}:{TYPE_MAP.get(p_type_int, '?')}"
                label_color = TYPE_COLOR_MAP.get(p_type_int, (255, 255, 255))

                text_pos_raw = optimized_positions[i]
                kp_canvas = (int(kp[0] + SIDEBAR_WIDTH), int(kp[1]))
                text_canvas = (int(text_pos_raw[0] + SIDEBAR_WIDTH), int(text_pos_raw[1]))

                draw_queue.append({
                    'kp': kp_canvas,
                    'text_pos': text_canvas,
                    'text': label_text,
                    'color': label_color
                })

        # Layer 1: 引线
        for item_d in draw_queue:
            cv2.line(canvas_bgr, item_d['kp'], item_d['text_pos'], item_d['color'], 1, cv2.LINE_AA)
            cv2.circle(canvas_bgr, item_d['kp'], 2, item_d['color'], -1)

        # Layer 2: 文字背景框
        overlay = canvas_bgr.copy()
        for item_d in draw_queue:
            (t_w, t_h), baseline = cv2.getTextSize(item_d['text'], cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            x, y = item_d['text_pos']
            box_x1 = x - t_w // 2 - 2
            box_y1 = y - t_h - 2
            box_x2 = x + t_w // 2 + 2
            box_y2 = y + 2
            cv2.rectangle(overlay, (box_x1, box_y1), (box_x2, box_y2), (255, 255, 255), -1)

        cv2.addWeighted(overlay, 0.7, canvas_bgr, 0.3, 0, canvas_bgr)

        # Layer 3: 文字本体
        for item_d in draw_queue:
            (t_w, t_h), _ = cv2.getTextSize(item_d['text'], cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            x, y = item_d['text_pos']
            text_org = (x - t_w // 2, y)
            cv2.putText(canvas_bgr, item_d['text'], text_org,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

        # --- 6. 绘制侧边栏信息 (核心修改处) ---
        ghost_indices = item['metrics']['ghost_indices']

        type_err_indices = []
        for i, (pt, gt) in enumerate(zip(pred_types, gt_types)):
            # 【修改】: 只要类型不相等，全部列入 Error
            # 这会自动包含 Ghost 情况 (例如: Pred=Normal[0], GT=Missing[2] -> 显示 "0(N->M)")
            if pt != gt:
                type_err_indices.append(f"{i}({TYPE_MAP.get(int(pt))}->{TYPE_MAP.get(int(gt))})")

        summary_lines = []
        summary_lines.append(f"Rank: {rank} | InstID: {target_inst_id}")
        summary_lines.append(f"ImgID: {item.get('img_id', 'N/A')}")
        summary_lines.append(f"AnnID: {item.get('ann_id', 'N/A')}")
        summary_lines.append(f"Metric: {metric_fmt(item['metrics'])}")
        summary_lines.append("-" * 35)
        # 依然保留 Ghost KPs 的独立展示，因为这是基于 Score 过滤的“严重”幻觉
        summary_lines.append(f"Ghost KPs (Score>{score_thr}):")
        if ghost_indices:
            summary_lines.append(str(ghost_indices))
        else:
            summary_lines.append("None")
        summary_lines.append("-" * 35)
        # Type Errs 现在包含所有类型错误
        summary_lines.append(f"All Type Errs ({len(type_err_indices)}):")
        if type_err_indices:
            # 3个一组换行，防止太长
            chunks = [type_err_indices[i:i + 3] for i in range(0, len(type_err_indices), 3)]
            for chunk in chunks:
                summary_lines.append(", ".join(chunk))
        else:
            summary_lines.append("None")

        text_y = 30
        for line in summary_lines:
            cv2.putText(canvas_bgr, line, (10, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
            text_y += 25

        # 保存
        m_val = metric_fmt(item['metrics'])
        fname = f"rank{rank:03d}_inst{target_inst_id}_{m_val}.jpg"
        out_path = os.path.join(folder, fname)
        cv2.imwrite(out_path, canvas_bgr)


def print_global_statistics(all_instances, score_thr):
    """
    计算并打印全局的 Type Error Rate 和 Ghost Rate
    """
    # 计数器
    total_kps_count = 0  # 所有关键点总数
    total_type_errors = 0  # 所有类型预测错误的点数

    total_gt_missing = 0  # 真值为 Missing (2) 的点总数
    total_ghosts_vis = 0  # 严重的幻觉点 (GT=2, Pred!=2, Score>Thr)

    total_gt_visible = 0  # 真值为 Visible/Prosthetic (0/1) 的点总数
    total_missed = 0  # 漏检 (GT!=2, Pred=2)

    for item in all_instances:
        # 获取数据
        target_inst_id = item['inst_id']
        gt_types = item['gt_types_numpy']

        # 预防性获取 pred_types
        if 'keypoint_types' in item['pred_all_numpy']:
            pred_types = item['pred_all_numpy']['keypoint_types'][target_inst_id]
        else:
            pred_types = np.zeros(len(gt_types), dtype=int)

        scores = item['pred_all_numpy']['keypoint_scores'][target_inst_id]

        # 1. 统计 Type Error (包含 Ghost, Miss, 以及 0<->1 混淆)
        # 只要预测类型和真值不一样，都算错
        type_mismatches = (pred_types != gt_types)
        total_type_errors += np.sum(type_mismatches)
        total_kps_count += len(gt_types)

        # 2. 统计 Ghost Rate (分母是 GT=Missing 的点)
        # 只有当 GT=Missing(2) 时，才有可能产生 Ghost
        missing_mask = (gt_types == 2)
        total_gt_missing += np.sum(missing_mask)

        # 这里为了严谨，统计的是“显性幻觉”：即模型不仅类型判错，而且置信度还很高
        # (GT=2) AND (Pred!=2) AND (Score > Thr)
        # 这与你 analyzer 中的 ghost_cnt 逻辑保持一致
        ghost_mask = (gt_types == 2) & (pred_types != 2) & (scores > score_thr)
        total_ghosts_vis += np.sum(ghost_mask)

        # 3. (可选) 统计 Miss Rate (漏检率)
        visible_mask = (gt_types != 2)
        total_gt_visible += np.sum(visible_mask)
        # GT!=2 却被预测为 2
        miss_mask = (gt_types != 2) & (pred_types == 2)
        total_missed += np.sum(miss_mask)

    # 计算比率
    type_error_rate = total_type_errors / total_kps_count if total_kps_count > 0 else 0
    ghost_rate = total_ghosts_vis / total_gt_missing if total_gt_missing > 0 else 0
    miss_rate = total_missed / total_gt_visible if total_gt_visible > 0 else 0

    print("\n" + "=" * 60)
    print("📊 Global Statistics Summary (全数据集统计)")
    print("=" * 60)
    print(f"Total Instances: {len(all_instances)}")
    print(f"Total Keypoints: {total_kps_count}")
    print("-" * 40)
    print(f"🔴 Global Type Error Rate: {type_error_rate:.2%} ({total_type_errors}/{total_kps_count})")
    print(f"   (定义: Pred_Type != GT_Type 的所有情况)")
    print("-" * 40)
    print(f"👻 Global Ghost Rate:      {ghost_rate:.2%} ({total_ghosts_vis}/{total_gt_missing})")
    print(f"   (定义: GT=Missing, Pred!=Missing, Score>{score_thr})")
    print("-" * 40)
    print(f"📉 Global Miss Rate:       {miss_rate:.2%} ({total_missed}/{total_gt_visible})")
    print(f"   (定义: GT=Visible, Pred=Missing)")
    print("=" * 60 + "\n")


def analyze_type_performance(all_instances):
    # 1. 收集所有的 GT 和 Pred
    y_true = []
    y_pred = []

    for item in all_instances:
        gt = item['gt_types_numpy']
        # 预防性获取 pred
        if 'keypoint_types' in item['pred_all_numpy']:
            pred = item['pred_all_numpy']['keypoint_types'][item['inst_id']]
        else:
            raise ValueError("Pred does not contain keypoint_types!")

        y_true.extend(gt)
        y_pred.extend(pred)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    target_names = ['Normal (0)', 'Prosthetic (1)', 'Missing (2)']

    # 2. 打印分类报告 (包含 Precision, Recall, F1-score)
    print("\n" + "=" * 60)
    print("🔬 详细分类性能分析 (Per-class Performance)")
    print("=" * 60)
    print(classification_report(y_true, y_pred, target_names=target_names, digits=4))

    # 3. 打印混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    print("-" * 30)
    print("🧩 混淆矩阵 (Confusion Matrix)")
    print("Row: GT, Col: Pred")
    print("-" * 30)
    print(f"{'':>15} {'Pred 0':>10} {'Pred 1':>10} {'Pred 2':>10}")
    for i, label in enumerate(target_names):
        print(f"{label:>15} {cm[i, 0]:>10d} {cm[i, 1]:>10d} {cm[i, 2]:>10d}")
    print("=" * 60 + "\n")

    return cm


def analyze_oks_detailed_groups(all_instances, dataset_sigmas):
    """
    三维对比分析：
    1. Standard Normal (0-16, Type 0): 普通肢体点
    2. Standard Prosthetic (0-16, Type 1): 假肢点 (替代了原肢体位置)
    3. Residual/Extra (17-24): 残肢点 (额外定义的点)
    """
    print("\n" + "=" * 70)
    print("📏 分组定位精度分析 (Group-wise Coordinate Precision)")
    print("=" * 70)

    # 初始化三组容器
    stats = {
        'normal_std': [],  # Indices 0-16, Type 0
        'prosthetic_std': [],  # Indices 0-16, Type 1
        'residual_ext': []  # Indices 17-24 (无论Type是0还是1，只要在这个范围都算残肢点)
    }

    # 检查 Sigmas 长度，防止越界
    # 如果 sigmas 只有 17 个，后续计算 17-24 会报错，这里做一个保护
    safe_sigmas = dataset_sigmas
    if len(safe_sigmas) < 25:
        print(f"⚠️ 警告: dataset_sigmas 长度为 {len(safe_sigmas)}，不足 25。")
        print("   将为索引 17-24 使用默认 Sigma (0.05) 进行计算，结果仅供参考。")
        # 补齐 sigmas
        pad_len = 25 - len(safe_sigmas)
        safe_sigmas = np.concatenate([safe_sigmas, np.full(pad_len, 0.05)])

    for item in all_instances:
        # 1. 准备数据
        inst_id = item['inst_id']
        gt_info = item['gt_info']

        # 获取 Area
        gt_kps_all = to_numpy(gt_info['keypoints'])
        if 'area' in gt_info:
            gt_areas = to_numpy(gt_info['area'])
            if gt_areas.ndim == 0: gt_areas = gt_areas[None]
        elif 'bbox' in gt_info:
            bboxes = to_numpy(gt_info['bbox'])
            if bboxes.ndim == 1: bboxes = bboxes[None, :]
            gt_areas = bboxes[:, 2] * bboxes[:, 3]
        else:
            gt_areas = np.ones(len(gt_kps_all))
        area = gt_areas[inst_id] if inst_id < len(gt_areas) else gt_areas[0]

        # 坐标与类型
        gt_kps = gt_kps_all[inst_id]
        gt_types = item['gt_types_numpy']
        pred_kps = item['pred_all_numpy']['keypoints'][inst_id]

        # 2. 逐点分类计算
        for k in range(len(gt_types)):
            t = int(gt_types[k])

            # Missing (2) 不参与坐标精度计算
            if t == 2:
                continue

            if len(gt_kps[k]) > 2 and gt_kps[k][2] == 0:
                continue

            # 计算单点 OKS
            dx = pred_kps[k][0] - gt_kps[k][0]
            dy = pred_kps[k][1] - gt_kps[k][1]
            dist_sq = dx ** 2 + dy ** 2
            sigma = safe_sigmas[k]
            e = dist_sq / (2 * area * (sigma ** 2) + 1e-9)
            oks = np.exp(-e)

            # --- 分组逻辑 ---
            if 0 <= k <= 16:
                # 标准骨架范围 (COCO 0-16)
                if t == 0:
                    stats['normal_std'].append(oks)
                elif t == 1:
                    stats['prosthetic_std'].append(oks)
            elif 17 <= k <= 24:
                # 扩展范围 (残肢/额外点)
                # 只要在这个 index 范围，不管它被标成 Normal 还是 Prosthetic，都归为残肢组
                stats['residual_ext'].append(oks)

    # 3. 打印统计结果
    groups = [
        ('Standard Normal (0-16)', 'normal_std'),
        ('Standard Pros.  (0-16)', 'prosthetic_std'),
        ('Residual Limb   (17-24)', 'residual_ext')
    ]

    print(f"{'Group Name':<25} | {'Avg OKS':<10} | {'Count':<8} | {'Evaluation'}")
    print("-" * 65)

    for name, key in groups:
        values = stats[key]
        if len(values) == 0:
            avg_val = 0.0
            count = 0
            comment = "N/A (No Samples)"
        else:
            avg_val = np.mean(values)
            count = len(values)
            if avg_val > 0.85:
                comment = "Excellent"
            elif avg_val > 0.70:
                comment = "Good"
            elif avg_val > 0.50:
                comment = "Fair"
            else:
                comment = "Poor"

        print(f"{name:<25} | {avg_val:.4f}     | {count:<8d} | {comment}")

    print("-" * 65)
    print("分析说明:")
    print("1. Standard Pros: 指那些占据了标准关节位置（如膝盖、脚踝）的假肢点。")
    print("2. Residual Limb: 指索引为 17-24 的自定义点（残肢末端等）。")
    print("=" * 70 + "\n")


def analyze_per_keypoint_contrast(all_instances, dataset_sigmas):
    """
    深度对比分析：逐个关键点对比 Normal vs Prosthetic 的 OKS。
    只针对 COCO 标准骨架 (0-16)，因为这些点才具备 "原本是肉 -> 变成了假肢" 的对比属性。
    """
    # COCO 关键点名称映射
    KP_NAMES = {
        0: "Nose", 1: "L_Eye", 2: "R_Eye", 3: "L_Ear", 4: "R_Ear",
        5: "L_Shoulder", 6: "R_Shoulder", 7: "L_Elbow", 8: "R_Elbow",
        9: "L_Wrist", 10: "R_Wrist", 11: "L_Hip", 12: "R_Hip",
        13: "L_Knee", 14: "R_Knee", 15: "L_Ankle", 16: "R_Ankle"
    }

    print("\n" + "=" * 85)
    print("⚔️  逐点对抗分析: Normal vs Prosthetic (Per-Keypoint Contrast)")
    print("=" * 85)

    # 数据结构: stats[kp_idx][type_id] = [oks1, oks2, ...]
    stats = {i: {0: [], 1: []} for i in range(17)}

    # 补齐 sigmas
    safe_sigmas = dataset_sigmas
    if len(safe_sigmas) < 17:
        safe_sigmas = np.concatenate([safe_sigmas, np.full(17 - len(safe_sigmas), 0.05)])

    for item in all_instances:
        inst_id = item['inst_id']
        gt_info = item['gt_info']

        # Area calculation
        gt_kps_all = to_numpy(gt_info['keypoints'])
        if 'area' in gt_info:
            gt_areas = to_numpy(gt_info['area'])
            if gt_areas.ndim == 0: gt_areas = gt_areas[None]
        elif 'bbox' in gt_info:
            bboxes = to_numpy(gt_info['bbox'])
            if bboxes.ndim == 1: bboxes = bboxes[None, :]
            gt_areas = bboxes[:, 2] * bboxes[:, 3]
        else:
            gt_areas = np.ones(len(gt_kps_all))
        area = gt_areas[inst_id] if inst_id < len(gt_areas) else gt_areas[0]

        gt_kps = gt_kps_all[inst_id]
        gt_types = item['gt_types_numpy']
        pred_kps = item['pred_all_numpy']['keypoints'][inst_id]

        # 遍历 0-16 号点
        for k in range(17):
            if k >= len(gt_types): break  # 保护

            t = int(gt_types[k])

            # 只关心 Normal(0) 和 Prosthetic(1)
            if t not in [0, 1]: continue

            # 过滤 Vis=0
            if len(gt_kps[k]) > 2 and gt_kps[k][2] == 0: continue

            # OKS Calculation
            dx = pred_kps[k][0] - gt_kps[k][0]
            dy = pred_kps[k][1] - gt_kps[k][1]
            dist_sq = dx ** 2 + dy ** 2
            sigma = safe_sigmas[k]
            e = dist_sq / (2 * area * (sigma ** 2) + 1e-9)
            oks = np.exp(-e)

            stats[k][t].append(oks)

    # --- 打印表格 ---
    # Header
    print(f"{'ID':<3} | {'Name':<12} | {'Normal OKS':<12} | {'Pros. OKS':<12} | {'Gap (P-N)':<10} | {'Pros. Count'}")
    print("-" * 85)

    # Rows
    for k in range(17):
        name = KP_NAMES.get(k, f"KP_{k}")

        # Normal Stats
        vals_0 = stats[k][0]
        avg_0 = np.mean(vals_0) if len(vals_0) > 0 else 0.0

        # Prosthetic Stats
        vals_1 = stats[k][1]
        avg_1 = np.mean(vals_1) if len(vals_1) > 0 else 0.0
        count_1 = len(vals_1)

        # Gap
        gap = avg_1 - avg_0

        # 格式化 Gap 颜色/符号
        # 如果样本太少，Gap 没有统计意义，标记一下
        if count_1 < 5:
            gap_str = " N/A (Low Data)"
        else:
            gap_str = f"{gap:+.4f}"
            if gap < -0.15:
                gap_str += " 🔻"  # 严重下降
            elif gap < -0.05:
                gap_str += " 📉"  # 轻微下降

        # 只打印那些至少有 Normal 数据的点 (防止空行)
        if len(vals_0) > 0 or len(vals_1) > 0:
            print(f"{k:<3} | {name:<12} | {avg_0:.4f}       | {avg_1:.4f}       | {gap_str:<15} | {count_1}")

    print("-" * 85)
    print("🔻: 性能严重下降 (Gap < -0.15) | 📉: 性能轻微下降 (Gap < -0.05)")
    print("=" * 85 + "\n")


def analyze_per_keypoint_classification(all_instances):
    """
    逐点分类性能分析 (全量版)：
    统计每个关键点 (0-16) 在 Normal (0), Prosthetic (1), Missing (2) 上的 Precision 和 Recall。
    """
    KP_NAMES = {
        0: "Nose", 1: "L_Eye", 2: "R_Eye", 3: "L_Ear", 4: "R_Ear",
        5: "L_Shoulder", 6: "R_Shoulder", 7: "L_Elbow", 8: "R_Elbow",
        9: "L_Wrist", 10: "R_Wrist", 11: "L_Hip", 12: "R_Hip",
        13: "L_Knee", 14: "R_Knee", 15: "L_Ankle", 16: "R_Ankle"
    }

    print("\n" + "=" * 130)  # 变长横线以容纳更多列
    print("🎯 逐点分类性能分析: Precision & Recall (包含 Missing)")
    print("=" * 130)

    # 1. 更新表头：增加 M-Prec 和 M-Rec
    # N=Normal, P=Prosthetic, M=Missing
    header = f"{'ID':<3} | {'Name':<10} | {'N-Prec':<7} {'N-Rec':<7} | {'P-Prec':<7} {'P-Rec':<9} | {'M-Prec':<7} {'M-Rec':<7} | {'P-Supp'}"
    print(header)
    print("-" * 130)

    # 数据收集
    storage = {k: {'gt': [], 'pred': []} for k in range(17)}
    for item in all_instances:
        inst_id = item['inst_id']
        gt_types = item['gt_types_numpy']

        if 'keypoint_types' in item['pred_all_numpy']:
            pred_types = item['pred_all_numpy']['keypoint_types'][inst_id]
        else:
            pred_types = np.zeros(len(gt_types), dtype=int)

        for k in range(17):
            if k >= len(gt_types): break
            storage[k]['gt'].append(gt_types[k])
            storage[k]['pred'].append(pred_types[k])

    # 2. 逐点计算并打印
    for k in range(17):
        y_true = np.array(storage[k]['gt'])
        y_pred = np.array(storage[k]['pred'])

        # 【关键修改 1】：labels 包含 0, 1, 2
        precision, recall, _, support = precision_recall_fscore_support(
            y_true, y_pred, labels=[0, 1, 2], zero_division=0
        )

        # 【关键修改 2】：解包 3 个值
        n_prec, p_prec, m_prec = precision[0], precision[1], precision[2]
        n_rec, p_rec, m_rec = recall[0], recall[1], recall[2]
        n_supp, p_supp, m_supp = support[0], support[1], support[2]

        name = KP_NAMES.get(k, f"KP_{k}")

        # 格式化显示
        # 假肢 Recall 低于 50% 依然加警告
        p_rec_str = f"{p_rec:.1%}"
        if p_supp > 0 and p_rec < 0.5:
            p_rec_str += "⚠️"

        p_prec_str = f"{p_prec:.1%}" if p_supp > 0 else "-"
        p_rec_display = p_rec_str if p_supp > 0 else "-"

        # Missing 通常数据量很大，直接显示
        m_prec_str = f"{m_prec:.1%}"
        m_rec_str = f"{m_rec:.1%}"

        # 【关键修改 3】：更新打印格式，加入 M 的列
        row = f"{k:<3} | {name:<10} | {n_prec:.1%}<7 {n_rec:.1%}<7 | {p_prec_str:<7} {p_rec_display:<9} | {m_prec_str:<7} {m_rec_str:<7} | {p_supp:<6}"
        # 注意：上面的 f-string 只是为了对齐，你可以直接用下面这种更稳的写法：
        print(
            f"{k:<3} | {name:<10} | {n_prec:.1%} {n_rec:.1%}   | {p_prec_str:<7} {p_rec_display:<9} | {m_prec_str:<7} {m_rec_str:<7} | {p_supp:<6}")

    print("-" * 130)
    print("⚠️ : P-Recall < 50% (假肢漏检严重)")
    print("=" * 130 + "\n")


def analyze_residual_detection(all_instances):
    """
    残肢点 (17-24) 专项检测分析：
    由于残肢点只有 0 (存在) 和 2 (不存在/Missing)，
    这里的 Recall 代表“检出率”，Precision 代表“抗幻觉能力”。
    """
    # 假设你的残肢点定义如下 (根据你的数据集定义修改名称)
    RESIDUAL_NAMES = {
        17: "Res_Arm_L", 18: "Res_Arm_R",
        19: "Res_Elb_L", 20: "Res_Elb_R",
        21: "Res_Leg_L", 22: "Res_Leg_R",
        23: "Res_Kne_L", 24: "Res_Kne_R"
    }

    print("\n" + "=" * 90)
    print("🦾 残肢点专项检测分析 (Residual Limb Detection: Existence vs Missing)")
    print("=" * 90)
    print(
        f"{'ID':<3} | {'Name':<12} | {'Recall (检出率)':<15} | {'Prec (抗幻觉)':<15} | {'Missed':<8} | {'Ghosts':<8} | {'Support'}")
    print("-" * 90)

    # 数据收集
    storage = {k: {'tp': 0, 'fp': 0, 'fn': 0, 'support': 0} for k in range(17, 25)}

    for item in all_instances:
        inst_id = item['inst_id']
        gt_types = item['gt_types_numpy']

        # 获取预测类型
        if 'keypoint_types' in item['pred_all_numpy']:
            pred_types = item['pred_all_numpy']['keypoint_types'][inst_id]
        else:
            pred_types = np.zeros(len(gt_types), dtype=int)

        # 遍历 17-24
        for k in range(17, 25):
            if k >= len(gt_types): break

            gt = int(gt_types[k])
            pd = int(pred_types[k])

            # 逻辑矩阵
            # 我们关注的是 "0" (存在) 这个类
            if gt == 0:
                storage[k]['support'] += 1
                if pd == 0:
                    storage[k]['tp'] += 1  # 找到了
                else:
                    storage[k]['fn'] += 1  # 漏了 (GT=0, Pred=2) -> Missed
            elif gt == 2:
                if pd != 2:  # 通常 pd=0
                    storage[k]['fp'] += 1  # 瞎画 (GT=2, Pred=0) -> Ghost

    # 计算并打印
    for k in range(17, 25):
        stats = storage[k]
        tp = stats['tp']
        fp = stats['fp']
        fn = stats['fn']
        support = stats['support']

        # Recall = TP / (TP + FN)
        if (tp + fn) > 0:
            rec = tp / (tp + fn)
            rec_str = f"{rec:.1%}"
            if rec < 0.5: rec_str += " ⚠️"
        else:
            rec_str = "-"

        # Precision = TP / (TP + FP)
        if (tp + fp) > 0:
            prec = tp / (tp + fp)
            prec_str = f"{prec:.1%}"
        else:
            prec_str = "-"  # 模型一个都没预测出来

        name = RESIDUAL_NAMES.get(k, f"Res_{k}")

        # 只打印有数据的行 (Support > 0 或 模型预测出了点)
        if support > 0 or (tp + fp) > 0:
            print(f"{k:<3} | {name:<12} | {rec_str:<15} | {prec_str:<15} | {fn:<8} | {fp:<8} | {support}")

    print("-" * 90)
    print("Missed: 真值有，模型没找到 (GT=0->Pred=2)")
    print("Ghosts: 真值无，模型瞎画了 (GT=2->Pred=0)")
    print("=" * 90 + "\n")

# ==========================================
# 6. 主流程 (保持不变)
# ==========================================
def main():
    cfg = Config.fromfile(CONFIG_FILE)
    init_default_scope(cfg.get('default_scope', 'mmpose'))

    if 'vis_backends' in cfg.visualizer: cfg.visualizer.vis_backends = []
    if 'save_dir' in cfg.visualizer: cfg.visualizer.save_dir = None

    dirs = {
        'ghost': os.path.join(OUT_DIR, '1_ghost_cases'),
        'std_bad': os.path.join(OUT_DIR, '2_standard_bad_oks'),
        'ld_bad': os.path.join(OUT_DIR, '3_ldpros_bad_oks'),
        'type_bad': os.path.join(OUT_DIR, '4_type_accuracy_bad')
    }
    for d in dirs.values(): os.makedirs(d, exist_ok=True)

    dataset = DATASETS.build(cfg.test_dataloader.dataset)
    print(f"Loading results from {RESULT_FILE}...")
    results = mmengine.load(RESULT_FILE)

    ds_sigmas = None
    if hasattr(dataset, 'metainfo') and 'sigmas' in dataset.metainfo:
        ds_sigmas = np.array(dataset.metainfo['sigmas'])

    analyzer = LDPoseAnalyzer(dataset_sigmas=ds_sigmas)
    visualizer = VISUALIZERS.build(cfg.visualizer)
    visualizer.dataset_meta = dataset.metainfo

    all_instances = []
    print("正在分析数据...")

    for img_idx, pred in enumerate(results):
        data_info = dataset.get_data_info(img_idx)
        gt_kps = to_numpy(data_info['keypoints'])
        img_id = data_info.get('img_id', data_info.get('image_id', 'N/A'))
        gt_types_batch = data_info.get('keypoint_types', None)
        if gt_types_batch is None:
            gt_types_batch = np.zeros((len(gt_kps), len(gt_kps[0])), dtype=int)
        else:
            gt_types_batch = to_numpy(gt_types_batch)
        if 'area' in data_info:
            gt_areas = to_numpy(data_info['area'])
            if gt_areas.ndim == 0: gt_areas = gt_areas[None]
        elif 'bbox' in data_info:
            bboxes = to_numpy(data_info['bbox'])
            if bboxes.ndim == 1: bboxes = bboxes[None, :]
            gt_areas = bboxes[:, 2] * bboxes[:, 3]
        else:
            gt_areas = np.ones(len(gt_kps))

        pred_instances = pred['pred_instances']
        pred_kps_all = to_numpy(pred_instances['keypoints'])
        pred_scores_all = to_numpy(pred_instances['keypoint_scores'])
        pred_types_all = to_numpy(pred_instances['keypoint_types']) if 'keypoint_types' in pred_instances else None

        for inst_id in range(len(pred_kps_all)):
            if inst_id >= len(gt_kps): break
            ann_id = data_info['id']
            p_inst = {
                'keypoints': pred_kps_all[inst_id],
                'keypoint_scores': pred_scores_all[inst_id],
                'keypoint_types': pred_types_all[inst_id] if pred_types_all is not None else None
            }
            cur_gt_types = gt_types_batch[inst_id] if gt_types_batch.ndim > 1 else gt_types_batch
            g_inst = {
                'keypoints': gt_kps[inst_id],
                'keypoint_types': cur_gt_types
            }
            cur_area = gt_areas[inst_id] if inst_id < len(gt_areas) else gt_areas[0]
            metrics = analyzer.analyze_instance(p_inst, g_inst, cur_area)
            all_instances.append({
                'img_idx': img_idx, 'img_path': data_info['img_path'], 'inst_id': inst_id,
                'img_id': img_id, 'ann_id': ann_id, 'metrics': metrics, 'gt_info': data_info,
                'pred_all_numpy': {'keypoints': pred_kps_all, 'keypoint_scores': pred_scores_all,
                                   'keypoint_types': pred_types_all},
                'gt_types_numpy': cur_gt_types
            })

    print("筛选 Ghost (Top 100)...")
    ghost_list = [x for x in all_instances if x['metrics']['ghost_cnt'] > 0]
    ghost_list.sort(key=lambda x: x['metrics']['ghost_cnt'], reverse=True)
    draw_enhanced(ghost_list[:100], dirs['ghost'], lambda m: f"ghost{m['ghost_cnt']}", visualizer)

    print("筛选 Standard OKS Worst (Top 100)...")
    std_list = sorted(all_instances, key=lambda x: x['metrics']['std_oks'])
    draw_enhanced(std_list[:100], dirs['std_bad'], lambda m: f"stdOKS{m['std_oks']:.3f}", visualizer)

    print("筛选 LDPros OKS Worst (Top 100)...")
    ld_list = sorted(all_instances, key=lambda x: x['metrics']['ld_oks'])
    draw_enhanced(ld_list[:100], dirs['ld_bad'], lambda m: f"ldOKS{m['ld_oks']:.3f}", visualizer)

    print("筛选 Type Accuracy Worst (Top 100)...")
    type_list = sorted(all_instances, key=lambda x: x['metrics']['type_acc'])
    draw_enhanced(type_list[:100], dirs['type_bad'], lambda m: f"Acc{m['type_acc']:.2f}", visualizer)

    print_global_statistics(all_instances, SCORE_THR)

    analyze_type_performance(all_instances)

    analyze_oks_detailed_groups(all_instances, ds_sigmas)

    analyze_per_keypoint_contrast(all_instances, ds_sigmas)

    analyze_per_keypoint_classification(all_instances)

    analyze_residual_detection(all_instances)

    print(f"完成！请查看 {OUT_DIR}")


if __name__ == '__main__':
    main()