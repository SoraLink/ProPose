import json
import os


def convert_single_file(input_json_path, output_json_path):
    print(f"正在处理: {input_json_path} \n  -> 输出至: {output_json_path}")

    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)

    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 1. 适配 LDPose 的 categories (25个点)
    # 强制读取 'categories'，如果没有直接引发 KeyError 报错
    for cat in data['categories']:
        cat['num_keypoints'] = 25
        ldpose_keypoints = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
            'L-Elbow-Res-Above', 'R-Elbow-Res-Above',
            'L-Elbow-Res-Below', 'R-Elbow-Res-Below',
            'L-Knee-Res-Above', 'R-Knee-Res-Above',
            'L-Knee-Res-Below', 'R-Knee-Res-Below'
        ]
        cat['keypoints'] = ldpose_keypoints

    # 2. 核心转换逻辑
    # 强制读取 'annotations'，如果没有直接引发 KeyError 报错
    annotations = data['annotations']
    converted_count = 0

    for ann in annotations:
        # 强制读取，缺失任何一个关键字段直接报错崩溃，不默默跳过
        old_kpts = ann['keypoints']
        old_types = ann['keypoint_types']
        ann_id = ann['id']

        # 严格校验 31 个点，如果维度不对给个明确警告
        if len(old_kpts) != 93 or len(old_types) != 31:
            print(f"  [警告] Annotation ID {ann_id} 的维度不对 (kpts:{len(old_kpts)}, types:{len(old_types)})，跳过。")
            continue

        new_kpts = []

        # 保留索引：0-16 (原生COCO) 和 23-30 (残肢端点)
        keep_indices = list(range(17)) + list(range(23, 31))

        for idx in keep_indices:
            x = old_kpts[idx * 3]
            y = old_kpts[idx * 3 + 1]
            v = old_kpts[idx * 3 + 2]

            t = old_types[idx]

            # 核心规则：如果 type 是 2 (Missing)，强制 vis = 0 且坐标归零
            if t == 2:
                v = 0

            # COCO 规范：不可见点的坐标设为 0
            if v == 0:
                x, y = 0.0, 0.0

            new_kpts.extend([x, y, v])

        # 更新 annotation 字典
        ann['keypoints'] = new_kpts

        # 处理完毕后，强制删除 keypoint_types 字段
        del ann['keypoint_types']

        # 同理，如果之前保存了 global_type_weights 等自定义字段，安全清理
        if 'global_type_weights' in ann:
            del ann['global_type_weights']

        converted_count += 1

    # 3. 保存
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f)
    print(f"  完成！成功转换了 {converted_count} 个有效实例。\n")


def convert_multiple_files(tasks):
    print(f"准备转换 {len(tasks)} 个文件...\n" + "=" * 50)

    for task in tasks:
        # 这里也是强制读取任务配置
        input_path = task['input']
        output_path = task['output']

        if not os.path.exists(input_path):
            raise FileNotFoundError(f"[报错] 找不到输入文件: {input_path}")

        convert_single_file(input_path, output_path)

    print("=" * 50 + "\n全部文件清洗并转换完毕！")


if __name__ == "__main__":
    CONVERT_TASKS = [
        {
            "input": "/home/sora/workspace/dataset/pros_final/train_final/train_final.json",
            "output": "/home/sora/workspace/dataset/pros_final/train_final/ldpose_train_25kpts.json"
        },
        {
            "input": "/home/sora/workspace/dataset/pros_final/test_final/test_final.json",
            "output": "/home/sora/workspace/dataset/pros_final/train_final/ldpose_test_25kpts.json"
        }
    ]

    convert_multiple_files(CONVERT_TASKS)