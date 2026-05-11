import argparse
import json
import os
import shutil
from collections import Counter, defaultdict
from copy import deepcopy


RESIDUAL_KPT_IDS = list(range(23, 31))

KPT_NAMES = {
    23: "L-Elbow-Res-Above",
    24: "R-Elbow-Res-Above",
    25: "L-Elbow-Res-Below",
    26: "R-Elbow-Res-Below",
    27: "L-Knee-Res-Above",
    28: "R-Knee-Res-Above",
    29: "L-Knee-Res-Below",
    30: "R-Knee-Res-Below",
}


def is_missing_keypoint(x, y, v, eps=1e-6):
    """A residual slot is treated as Absent if it has no valid location."""
    return v <= 0 or (abs(x) <= eps and abs(y) <= eps)


def fix_annotation(
    input_path,
    output_path,
    set_absent_vis=None,
    fix_visible_residual_to_bio=True,
):
    with open(input_path, "r") as f:
        data = json.load(f)

    if "annotations" not in data:
        raise ValueError("This does not look like a COCO annotation file: missing 'annotations'.")

    before_counts = defaultdict(Counter)
    after_counts = defaultdict(Counter)
    changed_counts = defaultdict(Counter)

    num_anns = 0
    num_changed_anns = 0
    num_missing_keypoint_types = 0

    for ann in data["annotations"]:
        num_anns += 1

        if "keypoints" not in ann:
            continue

        keypoints = ann["keypoints"]
        if len(keypoints) < 31 * 3:
            print(f"[Warning] ann_id={ann.get('id')} has fewer than 31 keypoints. Skipped.")
            continue

        if "keypoint_types" not in ann:
            # 如果没有 keypoint_types，就创建一个默认全 Bio 的数组
            ann["keypoint_types"] = [0] * 31
            num_missing_keypoint_types += 1

        if len(ann["keypoint_types"]) < 31:
            print(f"[Warning] ann_id={ann.get('id')} has fewer than 31 keypoint_types. Skipped.")
            continue

        old_types = deepcopy(ann["keypoint_types"])
        changed_this_ann = False

        for k in RESIDUAL_KPT_IDS:
            old_t = int(ann["keypoint_types"][k])
            before_counts[k][old_t] += 1

            x = keypoints[k * 3 + 0]
            y = keypoints[k * 3 + 1]
            v = keypoints[k * 3 + 2]

            if is_missing_keypoint(x, y, v):
                new_t = 2  # Absent / Missing

                # 可选：如果你的训练分类 loss 需要 v > 0 才监督 Absent，可以设成 2
                # 注意：mAP 评估前必须继续 clean absent keypoints，否则会把无意义坐标纳入 mAP
                if set_absent_vis is not None:
                    ann["keypoints"][k * 3 + 2] = set_absent_vis

            else:
                # residual keypoint 有真实位置，表示 physical stump，应为 Biological
                if fix_visible_residual_to_bio:
                    new_t = 0
                else:
                    new_t = old_t

            ann["keypoint_types"][k] = new_t
            after_counts[k][new_t] += 1

            if new_t != old_t:
                changed_counts[k][(old_t, new_t)] += 1
                changed_this_ann = True

        if changed_this_ann:
            num_changed_anns += 1

    if output_path == input_path:
        backup_path = input_path + ".bak"
        shutil.copy2(input_path, backup_path)
        print(f"[Backup] Original file backed up to: {backup_path}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True) if os.path.dirname(output_path) else None

    with open(output_path, "w") as f:
        json.dump(data, f)

    print("\n========== Summary ==========")
    print(f"Input : {input_path}")
    print(f"Output: {output_path}")
    print(f"Total annotations: {num_anns}")
    print(f"Annotations changed: {num_changed_anns}")
    print(f"Annotations missing keypoint_types originally: {num_missing_keypoint_types}")

    print("\n========== Residual Keypoint Type Counts ==========")
    for k in RESIDUAL_KPT_IDS:
        name = KPT_NAMES.get(k, f"kp_{k}")
        print(f"\n[{k}] {name}")
        print(f"  Before: {dict(before_counts[k])}")
        print(f"  After : {dict(after_counts[k])}")
        if changed_counts[k]:
            print(f"  Changed:")
            for (old_t, new_t), cnt in changed_counts[k].items():
                print(f"    {old_t} -> {new_t}: {cnt}")
        else:
            print("  Changed: none")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the original COCO annotation json.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to save the fixed annotation json. Can be the same as input.",
    )
    parser.add_argument(
        "--set-absent-vis",
        type=int,
        default=None,
        choices=[0, 1, 2],
        help=(
            "Optionally set visibility for residual Absent keypoints. "
            "Default: keep original visibility unchanged. "
            "Use 2 only if your classification loss needs v > 0 for Absent supervision."
        ),
    )
    parser.add_argument(
        "--keep-visible-residual-type",
        action="store_true",
        help=(
            "Do not force visible residual keypoints to Biological(0). "
            "Default behavior: visible residual keypoints are set to Bio(0)."
        ),
    )

    args = parser.parse_args()

    fix_annotation(
        input_path=args.input,
        output_path=args.output,
        set_absent_vis=args.set_absent_vis,
        fix_visible_residual_to_bio=not args.keep_visible_residual_type,
    )


if __name__ == "__main__":
    main()