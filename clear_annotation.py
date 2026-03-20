import json
import os


def filter_annotations_by_images(json_path, image_dir, output_path):
    print(f"开始加载 JSON: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 1. 扫描文件夹，获取所有物理图片的文件名（纯文件名，如 '0001.jpg'）
    existing_files = set(os.listdir(image_dir))
    print(f"文件夹中共有图片: {len(existing_files)} 张")

    # 2. 筛选有效的 images 列表
    valid_image_ids = set()
    new_images = []

    for img_obj in data['images']:
        # 使用 os.path.basename 提取纯文件名，防止 JSON 里的 file_name 带有 "images/xxx.jpg" 这种前缀
        pure_filename = os.path.basename(img_obj['file_name'])

        if pure_filename in existing_files:
            new_images.append(img_obj)
            valid_image_ids.add(img_obj['id'])

    print(f"JSON 中成功匹配到实体文件的图片记录: {len(new_images)} 条")

    # 3. 筛选 annotations（只保留 image_id 在有效集合中的）
    old_ann_count = len(data['annotations'])
    new_annotations = [
        ann for ann in data['annotations']
        if ann['image_id'] in valid_image_ids
    ]

    print(f"清理前 Annotation 数量: {old_ann_count}")
    print(f"清理后 Annotation 数量: {len(new_annotations)}")
    print(f"-> 删除了 {old_ann_count - len(new_annotations)} 条无实体图关联的标注。")

    # 4. 更新数据并保存
    data['images'] = new_images
    data['annotations'] = new_annotations

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f)

    print(f"完成！清洗后的文件已保存至: {output_path}")


if __name__ == "__main__":
    # --- 配置你的路径 ---
    CONFIG = {
        "input_json": "/home/sora/workspace/dataset/pros_final/train+crawl/train_final.json",
        "image_folder": "/home/sora/workspace/dataset/pros_final/train+crawl/images",
        "output_json": "/home/sora/workspace/dataset/pros_final/train+crawl/train_final_1.json"
    }

    filter_annotations_by_images(
        CONFIG["input_json"],
        CONFIG["image_folder"],
        CONFIG["output_json"]
    )