import functools
import torch
import cv2
import os
import numpy as np
from mmengine.config import Config
from mmengine.runner import Runner
from mmpose.apis import init_model

# 防止 pickle 安全报错
torch.load = functools.partial(torch.load, weights_only=False)

# ================= 配置区域 =================
CONFIG_FILE = './configs/body_2d_keypoint/topdown_heatmap/coco/DWPose_prosthetics.py'
CHECKPOINT = './work_dirs/DWPose_prosthetics/epoch_50.pth'

# 残肢点 ID
RESIDUAL_IDS = list(range(23, 31))
# 离谱误差阈值
HORROR_THR = 100


# ===========================================

def visualize_interactive():
    print(f"🚀 正在加载配置: {CONFIG_FILE}")
    cfg = Config.fromfile(CONFIG_FILE)
    cfg.work_dir = './work_dirs/debug_vis_temp'

    if 'test_cfg' in cfg.model:
        cfg.model.test_cfg['flip_test'] = False

    print(f"🧠 正在加载权重: {CHECKPOINT}")
    model = init_model(cfg, CHECKPOINT, device='cuda:0')
    model.eval()

    cfg.val_dataloader.batch_size = 1
    runner = Runner.from_cfg(cfg)
    data_loader = runner.val_dataloader

    print("\n" + "═" * 60)
    print("👀 裸输出模式 (No Filter Mode)")
    print("⚠️ 注意: 这里显示模型输出的所有点，无论置信度多低")
    print("🎮 [H] 搜寻残肢大误差 | [Space] 下一张 | [Q] 退出")
    print("-" * 60)
    print("🎨 图例:")
    print("   🌟 橙/蓝: 残肢点 (显示 ID 和 Score)")
    print("   ⚪ 绿/红: 普通点")
    print("   🔥 粗红线: 残肢离谱误差 (>100px)")
    print("═" * 60 + "\n")

    window_name = 'No Filter Debugger'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    data_iter = iter(data_loader)
    current_mode = 'normal'

    while True:
        try:
            batch = next(data_iter)
        except StopIteration:
            print("遍历结束。")
            break

        with torch.no_grad():
            results = model.test_step(batch)[0]

        img_path = batch['data_samples'][0].img_path
        img_name = os.path.basename(img_path)
        img = cv2.imread(img_path)
        if img is None: continue

        pred_instances = results.pred_instances
        gt_instances = results.gt_instances

        pred_kps = pred_instances.keypoints[0]
        pred_scores = pred_instances.keypoint_scores[0]
        gt_kps = gt_instances.keypoints[0]
        gt_vis = gt_instances.keypoints_visible[0]

        # --- Hunt Logic ---
        has_horror_residual = False
        for i in RESIDUAL_IDS:
            if i < len(gt_kps) and gt_vis[i] > 0:
                dist = np.linalg.norm(pred_kps[i] - gt_kps[i])
                if dist > HORROR_THR:
                    has_horror_residual = True
                    break

        if current_mode == 'hunt' and not has_horror_residual:
            print(f"Skipping {img_name}...", end='\r')
            continue

        if current_mode == 'hunt':
            print(f"\n🎯 Found horror case: {img_name}")
            current_mode = 'normal'

        # === 绘制逻辑 (无过滤) ===

        # 1. 绘制 GT
        for i, (gx, gy) in enumerate(gt_kps):
            if gt_vis[i] > 0:
                color = (0, 255, 0)  # Green
                marker = cv2.MARKER_CROSS
                size = 10
                thick = 1

                if i in RESIDUAL_IDS:
                    color = (0, 165, 255)  # Orange
                    marker = cv2.MARKER_STAR
                    size = 20
                    thick = 3

                cv2.drawMarker(img, (int(gx), int(gy)), color, markerType=marker, markerSize=size, thickness=thick)

        # 2. 绘制 Pred (移除 if pred_scores > thr 判断)
        for i, (px, py) in enumerate(pred_kps):

            # 颜色样式
            p_color = (0, 0, 255)  # Red (普通)
            radius = 4

            if i in RESIDUAL_IDS:
                p_color = (255, 0, 0)  # Blue (残肢)
                radius = 7

            # 直接画！不判断 score
            cv2.circle(img, (int(px), int(py)), radius, p_color, -1)

            # 残肢点：额外显示 ID 和 Score
            if i in RESIDUAL_IDS:
                # 格式: ID:24 sc:0.01
                info = f"{i} sc:{pred_scores[i]:.2f}"
                cv2.putText(img, info, (int(px) + 8, int(py) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, p_color, 2)

            # 3. 绘制连线
            if gt_vis[i] > 0:
                gx, gy = gt_kps[i]
                dist = np.linalg.norm([px - gx, py - gy])

                if i in RESIDUAL_IDS and dist > HORROR_THR:
                    # 🚨 离谱误差：粗红线
                    cv2.line(img, (int(gx), int(gy)), (int(px), int(py)), (0, 0, 255), 4)

                    # 显示距离
                    cv2.putText(img, f"Err:{int(dist)}", (int(px) + 10, int(py) + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                elif dist > 5:
                    # 普通误差：细灰线
                    cv2.line(img, (int(gx), int(gy)), (int(px), int(py)), (200, 200, 200), 1)

        # UI
        info_text = f"File: {img_name}"
        if has_horror_residual: info_text += " [HORROR ERROR]"

        cv2.putText(img, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img, "Showing RAW output (No Score Filter)", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (200, 200, 200), 1)

        cv2.imshow(window_name, img)
        key = cv2.waitKey(0)

        if key == ord('q') or key == 27:
            break
        elif key == ord('h'):
            print("🔍 Hunting for large residual errors...")
            current_mode = 'hunt'
        elif key == ord('s'):
            os.makedirs(cfg.work_dir, exist_ok=True)
            save_path = os.path.join(cfg.work_dir, f"raw_{img_name}")
            cv2.imwrite(save_path, img)
            print(f"Saved: {save_path}")

    cv2.destroyAllWindows()


if __name__ == '__main__':
    visualize_interactive()