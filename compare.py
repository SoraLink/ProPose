import pickle
import cv2
import numpy as np
import os
from tqdm import tqdm


# =========================================================
# 1. 独立实现的 OKS 计算公式
# =========================================================
def compute_oks(pred_kpts, gt_kpts, area, sigmas, visible):
    pred_kpts = np.array(pred_kpts)
    gt_kpts = np.array(gt_kpts)
    sigmas = np.array(sigmas)
    visible = np.array(visible) > 0

    if np.sum(visible) == 0:
        return 0.0

    distances = np.sum((pred_kpts - gt_kpts) ** 2, axis=1)
    variances = (sigmas * 2) ** 2
    e = distances / (2 * area * variances + 1e-9)
    oks = np.exp(-e)

    return np.sum(oks[visible]) / np.sum(visible)


# =========================================================
# 2. 官方 Sigma 定义
# =========================================================
my_sigmas = np.array([
    0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072,
    0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089,
    0.089, 0.089, 0.089, 0.089, 0.089, 0.089,
    0.072, 0.072, 0.062, 0.062, 0.087, 0.087, 0.089, 0.089
])


# =========================================================
# 3. 可视化和画图逻辑 (双重绝对过滤 + Bbox版)
# =========================================================
def draw_bbox(img, bbox, color, thickness=2):
    """画 Bounding Box 的辅助函数 (适配 MMPose pkl 的 x1, y1, x2, y2 格式)"""
    if bbox is None or len(bbox) < 4:
        return img

    x1, y1, x2, y2 = bbox[:4]
    pt1 = (int(x1), int(y1))
    pt2 = (int(x2), int(y2)) # 直接使用 x2, y2，不再加上宽高

    cv2.rectangle(img, pt1, pt2, color, thickness)
    return img


def draw_pose(img, kpts, color, types, valid_mask, radius=5):
    """严格过滤 Missing 和 vis=0 的点"""
    for i, (x, y) in enumerate(kpts):
        if types[i] == 2:
            continue
        if not valid_mask[i]:
            continue
        if x <= 0 or y <= 0:
            continue

        if 17 <= i <= 30:
            cv2.circle(img, (int(x), int(y)), radius + 2, color, 2)
        else:
            cv2.circle(img, (int(x), int(y)), radius, color, -1)
    return img


def visualize_by_oks(pkl_a, pkl_b, sigmas, data_root, output_dir='oks_vis_results_2', threshold=0.05):
    os.makedirs(os.path.join(output_dir, 'Contrast_Wins_AccUp'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'Baseline_Wins_APDown'), exist_ok=True)

    print(f"⏳ 正在加载结果进内存...")
    with open(pkl_a, 'rb') as f:
        data_a_raw = pickle.load(f)
    with open(pkl_b, 'rb') as f:
        data_b_raw = pickle.load(f)

    def extract_list(raw_data):
        if isinstance(raw_data, list): return raw_data
        if isinstance(raw_data, dict):
            for k, v in raw_data.items():
                if isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
                    return v
        return []

    data_a = extract_list(data_a_raw)
    data_b = extract_list(data_b_raw)

    count_b_wins, count_a_wins = 0, 0

    print(f"🔍 开始逐张、逐人对比...")

    for item_a, item_b in tqdm(zip(data_a, data_b), total=min(len(data_a), len(data_b)), desc="📊 进度"):
        raw_img_path = item_a.get('img_path', '')

        if not raw_img_path or raw_img_path != item_b.get('img_path', ''):
            continue

        img_basename = os.path.basename(raw_img_path)

        img_path = None
        if os.path.exists(raw_img_path):
            img_path = raw_img_path
        elif os.path.exists(os.path.join(data_root, raw_img_path)):
            img_path = os.path.join(data_root, raw_img_path)
        elif os.path.exists(os.path.join(data_root, 'test_final/images', img_basename)):
            img_path = os.path.join(data_root, 'test_final/images', img_basename)

        if img_path is None:
            continue

        num_instances = len(item_a['gt_instances']['keypoints'])
        for n in range(num_instances):
            gt = item_a['gt_instances']['keypoints'][n]
            # 这里提取出来的 valid 就是 GT 中的 visibility
            valid = item_a['gt_instances']['keypoints_visible'][n] > 0

            pred_a = item_a['pred_instances']['keypoints'][n]
            pred_b = item_b['pred_instances']['keypoints'][n]

            gt_types = item_a['gt_instances']['keypoint_types'][n]

            pred_types_a_all = item_a['pred_instances'].get('keypoint_types', [np.zeros(31)] * num_instances)
            pred_types_a = pred_types_a_all[n] if len(pred_types_a_all) > n else np.zeros(31)

            pred_types_b_all = item_b['pred_instances'].get('keypoint_types', [np.zeros(31)] * num_instances)
            pred_types_b = pred_types_b_all[n] if len(pred_types_b_all) > n else np.zeros(31)

            # --- 获取 BBox 和计算 Area ---
            gt_bbox = None
            if 'bboxes' in item_a['gt_instances'] and len(item_a['gt_instances']['bboxes']) > n:
                gt_bbox = item_a['gt_instances']['bboxes'][n]
                # 按照 x1, y1, x2, y2 格式正确计算面积：(x2 - x1) * (y2 - y1)
                area = (gt_bbox[2] - gt_bbox[0]) * (gt_bbox[3] - gt_bbox[1])
            else:
                area = (np.max(gt[valid, 0]) - np.min(gt[valid, 0])) * (np.max(gt[valid, 1]) - np.min(gt[valid, 1]))
            area = max(area, 1.0)

            # 安全提取预测的 bbox
            pred_a_bbox = item_a['pred_instances']['bboxes'][n] if 'bboxes' in item_a['pred_instances'] and len(
                item_a['pred_instances']['bboxes']) > n else None
            pred_b_bbox = item_b['pred_instances']['bboxes'][n] if 'bboxes' in item_b['pred_instances'] and len(
                item_b['pred_instances']['bboxes']) > n else None

            eval_valid = valid.copy()
            eval_valid[gt_types == 2] = False

            oks_a = compute_oks(pred_a, gt, area, sigmas, eval_valid)
            oks_b = compute_oks(pred_b, gt, area, sigmas, eval_valid)
            diff = oks_b - oks_a

            if abs(diff) < threshold: continue

            img = cv2.imread(img_path)

            # 🌟 画 GT：先画框，再画点
            img_gt = img.copy()
            img_gt = draw_bbox(img_gt, gt_bbox, (255, 255, 255))
            img_gt = draw_pose(img_gt, gt, (255, 255, 255), types=gt_types, valid_mask=valid)

            # 🌟 画 Baseline Pred_A：先画框，再画点
            img_a = img.copy()
            img_a = draw_bbox(img_a, pred_a_bbox, (0, 0, 255))
            img_a = draw_pose(img_a, pred_a, (0, 0, 255), types=gt_types, valid_mask=valid)

            # 🌟 画 Finetune Pred_B：先画框，再画点
            img_b = img.copy()
            img_b = draw_bbox(img_b, pred_b_bbox, (0, 255, 0))
            img_b = draw_pose(img_b, pred_b, (0, 255, 0), types=gt_types, valid_mask=valid)

            combined = np.hstack([img_gt, img_a, img_b])

            cv2.putText(combined, "Ground Truth", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(combined, f"Baseline CB | OKS: {oks_a:.3f}", (img.shape[1] + 20, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 255), 2)
            cv2.putText(combined, f"Finetune (Ours) | OKS: {oks_b:.3f}", (img.shape[1] * 2 + 20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            file_name = f'diff_{abs(diff):.3f}_inst{n}_{img_basename}'

            if diff >= threshold:
                cv2.imwrite(os.path.join(output_dir, 'Contrast_Wins_AccUp', file_name), combined)
                count_b_wins += 1
            else:
                cv2.imwrite(os.path.join(output_dir, 'Baseline_Wins_APDown', file_name), combined)
                count_a_wins += 1

    print(f"\n==============================")
    print(f"处理完成！请去 {output_dir} 查看。")
    print(f"==============================\n")


if __name__ == '__main__':
    # 改回你实际使用的文件名
    pkl_baseline = 'result_CB.pkl'
    pkl_ours = 'result_finetune.pkl'
    root_dir = '/home/sora/workspace/dataset/pros_final'
    visualize_by_oks(pkl_baseline, pkl_ours, my_sigmas, data_root=root_dir, threshold=0.05)