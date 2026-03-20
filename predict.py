import cv2
import os
import numpy as np
import torch
import functools
from collections import OrderedDict

# ==========================================
# 修复 PyTorch 加载问题
# ==========================================
original_torch_load = torch.load
torch.load = functools.partial(original_torch_load, weights_only=False)

from mmpose.apis import inference_topdown, init_model


def main():
    # ==========================================
    # 1. 多模型配置 (在这里添加你的4个模型)
    # ==========================================
    models_config = [
        {
            'name': 'VIT',
            'config': 'configs/body_2d_keypoint/topdown_heatmap/coco/VIT_L_prosthetics_combined_loss_finetune.py',
            'checkpoint': 'work_dirs/VIT_L_prosthetics_combined_loss_finetune/epoch_1.pth'
        },
        {
            'name': 'SWIN',
            'config': 'configs/body_2d_keypoint/topdown_heatmap/coco/Swin-l-256x192.py',
            'checkpoint': 'work_dirs/Swin-l-256x192/epoch_50.pth'
        },
        {
            'name': 'YOLOPOSE',
            'config': 'configs/body_2d_keypoint/yoloxpose/coco/yoloxpose_l-640.py',
            'checkpoint': 'work_dirs/yoloxpose_l-640/epoch_100.pth'
        },
        {
            'name': 'RTM',
            'config': 'configs/body_2d_keypoint/rtmpose/coco/rtmpose-l-CB-loss.py',
            'checkpoint': 'work_dirs/rtmpose-l-CB-loss/epoch_50.pth'
        }
    ]

    img_dir = 'protocol'  # 存放那5张图的文件夹
    base_out_dir = 'qualitative_comparison'  # 总输出目录

    # 可视化配置
    POINT_COLOR_NORMAL = (255, 255, 0);
    POINT_COLOR_PROSTHETIC = (255, 0, 255);
    POINT_COLOR_STUMP = (0, 255, 255)
    LINE_COLOR_NORMAL = (150, 150, 0);
    LINE_COLOR_PROSTHETIC = (150, 0, 150);
    LINE_COLOR_STUMP = (0, 150, 150)
    SCORE_THR = 0.6
    MAX_IMAGE_SIZE = 1080

    SKELETON_PAIRS = [
        (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (11, 12), (5, 11), (6, 12),
        (11, 13), (12, 14), (13, 15), (14, 16), (0, 1), (0, 2), (1, 3), (2, 4),
        (9, 17), (10, 18), (15, 19), (15, 21), (16, 20), (16, 22)
    ]

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # 准备图片列表
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    img_names = [f for f in os.listdir(img_dir) if f.lower().endswith(valid_extensions)]

    # ==========================================
    # 2. 交互环节：先对所有图片画一次框并记录
    # ==========================================
    # 为了保证对比公平，4个模型必须使用完全相同的 BBox
    img_bboxes = {}
    print("--- 步骤 1: 请为所有图片统一画框 (所有模型将共用此框) ---")
    for img_name in img_names:
        img = cv2.imread(os.path.join(img_dir, img_name))
        if img is None: continue

        # 统一缩放逻辑以便画框
        h, w = img.shape[:2]
        scale = 1.0
        if max(h, w) > MAX_IMAGE_SIZE:
            scale = MAX_IMAGE_SIZE / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)))

        cv2.namedWindow("Draw BBoxes", cv2.WINDOW_NORMAL)
        rois = cv2.selectROIs("Draw BBoxes", img, False, False)
        cv2.destroyAllWindows()

        bboxes = []
        for (rx, ry, rw, rh) in rois:
            # 还原到原图尺寸并扩宽 1.25 倍
            cx, cy = (rx + rw / 2) / scale, (ry + rh / 2) / scale
            nw, nh = (rw / scale) * 1.25, (rh / scale) * 1.25
            bboxes.append([cx - nw / 2, cy - nh / 2, cx + nw / 2, cy + nh / 2])
        img_bboxes[img_name] = np.array(bboxes)

    # ==========================================
    # 3. 核心循环：遍历模型预测
    # ==========================================
    for m_info in models_config:
        m_name = m_info['name']
        print(f"\n>>> 正在加载模型: {m_name}")

        # 初始化模型
        model = init_model(m_info['config'], m_info['checkpoint'], device=device)

        # 创建模型专属文件夹
        m_out_dir = os.path.join(base_out_dir, m_name)
        os.makedirs(m_out_dir, exist_ok=True)

        for img_name in img_names:
            img = cv2.imread(os.path.join(img_dir, img_name))
            bboxes = img_bboxes.get(img_name, [])
            if len(bboxes) == 0: continue

            # 推理
            results = inference_topdown(model, img, bboxes, bbox_format='xyxy')

            # 绘制逻辑 (保持你之前的配色方案)
            h, w = img.shape[:2]
            dr = max(2, int(min(h, w) * 0.008))
            dt = max(1, int(min(h, w) * 0.002))

            for data_sample in results:
                pred = data_sample.pred_instances
                for p_idx in range(len(pred.keypoints)):
                    kpts = pred.keypoints[p_idx]
                    scores = pred.keypoint_scores[p_idx]
                    types = pred.keypoint_types[p_idx]

                    # 画线
                    for idx_a, idx_b in SKELETON_PAIRS:
                        if idx_a < len(kpts) and idx_b < len(kpts):
                            if scores[idx_a] > SCORE_THR and scores[idx_b] > SCORE_THR:
                                t_a, t_b = int(types[idx_a]), int(types[idx_b])
                                if t_a == 2 or t_b == 2: continue

                                color = LINE_COLOR_NORMAL
                                if (24 <= idx_a <= 31) or (24 <= idx_b <= 31):
                                    color = LINE_COLOR_STUMP
                                elif t_a == 1 or t_b == 1:
                                    color = LINE_COLOR_PROSTHETIC

                                cv2.line(img, (int(kpts[idx_a][0]), int(kpts[idx_a][1])),
                                         (int(kpts[idx_b][0]), int(kpts[idx_b][1])), color, dt, cv2.LINE_AA)
                    # 画点
                    for i in range(len(kpts)):
                        if scores[i] > SCORE_THR and int(types[i]) != 2:
                            t = int(types[i])
                            color = POINT_COLOR_NORMAL
                            if 24 <= i <= 31:
                                color = POINT_COLOR_STUMP
                            elif t == 1:
                                color = POINT_COLOR_PROSTHETIC
                            cv2.circle(img, (int(kpts[i][0]), int(kpts[i][1])), dr, color, -1, cv2.LINE_AA)

            # 保存到对应模型文件夹
            cv2.imwrite(os.path.join(m_out_dir, img_name), img)
            print(f"  [Saved] {m_name} -> {img_name}")

    print(f"\n✅ 所有对比图已生成在 '{base_out_dir}' 目录下！")


if __name__ == '__main__':
    main()