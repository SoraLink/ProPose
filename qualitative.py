import torch
import numpy

# 1. Allow the specific numpy reconstruction function mentioned in the error
try:
    torch.serialization.add_safe_globals([numpy.core.multiarray._reconstruct])
except AttributeError:
    pass

# 2. Monkey-patch torch.load to revert back to PyTorch 2.5 behavior
_original_torch_load = torch.load

def _legacy_torch_load(*args, **kwargs):
    # Force weights_only to False if it hasn't been explicitly set
    kwargs.setdefault('weights_only', False)
    return _original_torch_load(*args, **kwargs)

torch.load = _legacy_torch_load

import os
from mmpose.apis import MMPoseInferencer


def main():
    # 1. Define the images you selected for the paper
    image_paths = [
        './qual_images/123.jpg',
        './qual_images/000132_jpg.jpg',
        './qual_images/234.jpg'
    ]

    output_base_dir = 'qualitative_results'
    os.makedirs(output_base_dir, exist_ok=True)

    # 2. Define your models: { "Display Name": ("path/to/config.py", "path/to/checkpoint.pth") }
    # Update these paths to match your local setup
    models = {
        'YOLOX-Pose-L': (
            'configs/body_2d_keypoint/yoloxpose/coco/yoloxpose_l_8xb32-300e_coco-640.py',
            'models/yoloxpose.pth'
        ),
        'Swin-L': (
            'configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_swin-l-p4-w7_8xb32-210e_coco-256x192.py',
            'models/swin.pth'
        ),
        'RTMPose-L': (
            'configs/body_2d_keypoint/rtmpose/coco/rtmpose-l_8xb256-420e_coco-256x192.py',
            'models/rtmpose.pth'
        ),
        'ViTPose-L': (
            'configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-large_8xb64-210e_coco-256x192.py',
            'models/ViTPose.pth'
        ),
        'Ours_ProPose': (
            'configs/body_2d_keypoint/topdown_heatmap/coco/VIT_L_prosthetics_combined_loss_finetune.py',
            'work_dirs/VIT_L_prosthetics_combined_loss_finetune/epoch_1.pth'
        )
    }

    # 3. Loop through models and run inference
    for model_name, (config_path, checkpoint_path) in models.items():
        print(f"\n[{model_name}] Initializing inferencer...")

        # Initialize the inferencer for this specific model
        # Note: If your images have multiple people, you might want to add a det_model to the inferencer
        inferencer = MMPoseInferencer(
            pose2d=config_path,
            pose2d_weights=checkpoint_path,
            det_model='demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py',
            det_weights='https://download.openmmlab.com/mmpose/v1.0/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth',
            device='cuda:0'  # Change if using a different GPU
        )

        # Create a specific output folder for this model
        model_out_dir = os.path.join(output_base_dir, model_name)
        os.makedirs(model_out_dir, exist_ok=True)

        for img_path in image_paths:
            print(f"[{model_name}] Processing {os.path.basename(img_path)}...")

            # Run inference and save visualization
            # The inferencer returns a generator, so we iterate through it
            result_generator = inferencer(
                img_path,
                show=False,
                out_dir=model_out_dir,
                thickness=3,  # Adjust skeleton line thickness for paper visibility
                radius=4  # Adjust keypoint dot radius for paper visibility
            )

            # Execute the generator
            for result in result_generator:
                pass

    print(f"\nAll done! Visualizations saved to {output_base_dir}/")


if __name__ == '__main__':
    main()