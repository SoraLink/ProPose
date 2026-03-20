import cv2
import os
import json
import numpy as np


def main():
    # ==========================================
    # 1. 基础路径配置 (请修改为你的实际路径)
    # ==========================================
    img_dir = 'protocol'  # 存放图片的文件夹
    out_dir = 'protocol_gt_output'  # 保存画好GT的输出文件夹
    DATA_ROOT = '/home/sora/workspace/dataset/pros_final'
    annotation_file = os.path.join(DATA_ROOT, 'train_final/train_final.json')

    os.makedirs(out_dir, exist_ok=True)

    # ==========================================
    # 2. 颜色与样式配置 (BGR格式)
    # ==========================================
    POINT_COLOR_NORMAL = (255, 255, 0)  # 亮青色
    POINT_COLOR_PROSTHETIC = (255, 0, 255)  # 亮洋红
    POINT_COLOR_STUMP = (0, 255, 255)  # 亮黄色

    LINE_COLOR_NORMAL = (150, 150, 0)  # 暗青色
    LINE_COLOR_PROSTHETIC = (150, 0, 150)  # 暗洋红
    LINE_COLOR_STUMP = (0, 150, 150)  # 暗黄色

    MAX_IMAGE_SIZE = 1080
    POINT_RADIUS_RATIO = 0.008
    LINE_THICKNESS_RATIO = 0.002

    # ==========================================
    # 3. 定义骨架连线
    # ==========================================
    SKELETON_PAIRS = [
        (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (11, 12), (5, 11), (6, 12),
        (11, 13), (12, 14), (13, 15), (14, 16),
        (0, 1), (0, 2), (1, 3), (2, 4),
        (9, 17), (10, 18),
        (15, 19), (15, 21),
        (16, 20), (16, 22),
    ]

    # ==========================================
    # 4. 读取标注文件并建立索引映射
    # ==========================================
    print(f"正在读取标注文件: {annotation_file} ...")
    with open(annotation_file, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)

    # 建立 "图片名称 -> image_id" 的映射字典
    # 注意：为了防止路径干扰，我们用 os.path.basename 只取文件名进行匹配
    name_to_id = {os.path.basename(img_info['file_name']): img_info['id']
                  for img_info in coco_data['images']}

    # 建立 "image_id -> 包含的所有人的标注列表" 的映射字典
    from collections import defaultdict
    id_to_annotations = defaultdict(list)
    for ann in coco_data['annotations']:
        id_to_annotations[ann['image_id']].append(ann)

    # ==========================================
    # 5. 遍历本地图片并画图
    # ==========================================
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    img_names = [f for f in os.listdir(img_dir) if f.lower().endswith(valid_extensions)]

    print(f"共找到 {len(img_names)} 张图片，开始绘制真实标注(Ground Truth)...")

    for img_name in img_names:
        # 匹配图片 ID
        image_id = name_to_id.get(img_name)
        if image_id is None:
            print(f"⚠️ 警告: 标注文件中未找到图片 {img_name}，跳过。")
            continue

        # 获取这张图片里所有人的标注
        anns = id_to_annotations.get(image_id, [])
        if not anns:
            print(f"⚠️ 警告: 图片 {img_name} 没有对应的人体标注，跳过。")
            continue

        img_path = os.path.join(img_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        h, w = img.shape[:2]

        # 计算缩放比例 (如果图片太大)
        scale = 1.0
        if max(h, w) > MAX_IMAGE_SIZE:
            scale = MAX_IMAGE_SIZE / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)))
            h, w = img.shape[:2]

        dynamic_radius = max(2, int(min(h, w) * POINT_RADIUS_RATIO))
        dynamic_line_thickness = max(1, int(min(h, w) * LINE_THICKNESS_RATIO))

        # ==========================================
        # 6. 开始遍历画每个人
        # ==========================================
        for ann in anns:
            # 提取 keypoints，COCO格式通常是 [x1, y1, v1, x2, y2, v2, ...]
            kpts_flat = ann.get('keypoints', [])
            if not kpts_flat:
                continue

            # 转成 N x 3 的矩阵: [[x, y, v], [x, y, v], ...]
            kpts = np.array(kpts_flat).reshape(-1, 3)

            # 提取 keypoint_types
            kpt_types = ann.get('keypoint_types', [])
            # 防御性编程：如果JSON里不小心漏了 type，默认全给 0 (Normal)
            if not kpt_types:
                kpt_types = [0] * len(kpts)

            # --- 6.1 先画深色细线 ---
            for pair in SKELETON_PAIRS:
                idx_a, idx_b = pair
                if idx_a >= len(kpts) or idx_b >= len(kpts):
                    continue

                x_a, y_a, v_a = kpts[idx_a]
                x_b, y_b, v_b = kpts[idx_b]
                type_a = kpt_types[idx_a]
                type_b = kpt_types[idx_b]

                # COCO 格式中 v>0 代表有标注 (v=1是被遮挡，v=2是可见，我们通常全画)
                if v_a > 0 and type_a != 2 and v_b > 0 and type_b != 2:

                    if (24 <= idx_a <= 31) or (24 <= idx_b <= 31):
                        line_color = LINE_COLOR_STUMP
                    elif type_a == 1 or type_b == 1:
                        line_color = LINE_COLOR_PROSTHETIC
                    else:
                        line_color = LINE_COLOR_NORMAL

                    # 【关键】因为图片可能被等比例缩放了，所以坐标也必须乘以 scale
                    pos_a = (int(x_a * scale), int(y_a * scale))
                    pos_b = (int(x_b * scale), int(y_b * scale))

                    cv2.line(img, pos_a, pos_b, line_color, dynamic_line_thickness, cv2.LINE_AA)

            # --- 6.2 再画高亮圆点 ---
            for i in range(len(kpts)):
                x, y, v = kpts[i]
                kpt_type = kpt_types[i]

                if v > 0 and kpt_type != 2:

                    if 24 <= i <= 31:
                        pt_color = POINT_COLOR_STUMP
                    elif kpt_type == 1:
                        pt_color = POINT_COLOR_PROSTHETIC
                    else:
                        pt_color = POINT_COLOR_NORMAL

                    pos = (int(x * scale), int(y * scale))
                    cv2.circle(img, pos, dynamic_radius, pt_color, thickness=-1, lineType=cv2.LINE_AA)

        # ==========================================
        # 7. 在底部拼接 Legend 区域
        # ==========================================
        legend_h = 60  # 稍微加高一点点，看起来不拥挤
        legend_bar = np.zeros((legend_h, w, 3), dtype=np.uint8)

        legend_items = [
            ("Biological", POINT_COLOR_NORMAL),
            ("Residual", POINT_COLOR_STUMP),
            ("Prosthetic", POINT_COLOR_PROSTHETIC)
        ]

        # 更加精确的平铺计算
        section_w = w // len(legend_items)
        for i, (label, color) in enumerate(legend_items):
            # 这里的 offset 是为了让“圆点+文字”作为一个整体在 section 内居中
            offset_x = i * section_w + (section_w // 6)
            text_y = 38

            # 画圆点
            cv2.circle(legend_bar, (offset_x, text_y - 8), 8, color, -1, lineType=cv2.LINE_AA)
            # 写文字 (缩减了字体大小到 0.6，防止窄图显示不全)
            cv2.putText(legend_bar, label, (offset_x + 25, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        # 垂直拼接
        final_img = cv2.vconcat([img, legend_bar])
        # 保存结果
        out_file = os.path.join(out_dir, img_name)
        cv2.imwrite(out_file, final_img)
        print(f'已处理并保存(含底部图例): {out_file}')

    print(f'\n全部完成！Ground Truth 标注已保存在 {out_dir} 文件夹中。')


if __name__ == '__main__':
    main()