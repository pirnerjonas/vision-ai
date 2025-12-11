from pathlib import Path

import numpy as np
import supervision as sv
import torch
from dataset import YOLOSegmentationDataset
from PIL import Image
from transformers import AutoImageProcessor

# Configuration
CONFIG = {
    "dataset_path": "../datasets/yolo/crack",
    "split": "train",  # Can be "train", "valid", or "test"
    "output_dir": "./visualizations",
    "num_samples": 10,  # Number of samples to visualize
    "show_augmentations": True,  # Show original vs augmented side by side
}


def visualize():
    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load image processor
    image_processor = AutoImageProcessor.from_pretrained(
        "facebook/mask2former-swin-tiny-coco-instance",
        do_resize=True,
        size={"height": 512, "width": 512},
    )

    # Load datasets (original and augmented if requested)
    dataset = YOLOSegmentationDataset(
        CONFIG["dataset_path"],
        split=CONFIG["split"],
        image_processor=image_processor,
        augment=False,
    )

    dataset_aug = None
    if CONFIG["show_augmentations"]:
        dataset_aug = YOLOSegmentationDataset(
            CONFIG["dataset_path"],
            split=CONFIG["split"],
            image_processor=image_processor,
            augment=True,
        )

    # Annotator for visualization
    mask_annotator = sv.MaskAnnotator()

    print(
        f"Visualizing {min(CONFIG['num_samples'], len(dataset))} samples "
        f"from {CONFIG['split']} set..."
    )

    for idx in range(min(CONFIG["num_samples"], len(dataset))):
        # Get original sample
        sample = dataset[idx]
        pixel_values = sample["pixel_values"]
        mask_labels = sample["mask_labels"]
        class_labels = sample["class_labels"]

        # Denormalize image (reverse ImageNet normalization)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image = pixel_values * std + mean
        image = (image * 255).clamp(0, 255).permute(1, 2, 0).numpy().astype(np.uint8)
        image = np.ascontiguousarray(image)

        # Check if there are any instances
        if len(mask_labels) > 0:
            masks = mask_labels.numpy().astype(bool)
            print(f"  Sample {idx}: {len(mask_labels)} instances")

            detections = sv.Detections(
                xyxy=sv.mask_to_xyxy(masks=masks),
                mask=masks,
                class_id=class_labels.numpy(),
            )
            annotated_image = mask_annotator.annotate(
                scene=image.copy(), detections=detections
            )
        else:
            print(f"  Sample {idx}: No instances (background only)")
            annotated_image = image

        # If showing augmentations, get augmented version and concatenate
        if CONFIG["show_augmentations"] and dataset_aug is not None:
            sample_aug = dataset_aug[idx]
            pixel_values_aug = sample_aug["pixel_values"]
            mask_labels_aug = sample_aug["mask_labels"]
            class_labels_aug = sample_aug["class_labels"]

            # Denormalize augmented image
            image_aug = pixel_values_aug * std + mean
            image_aug = (
                (image_aug * 255)
                .clamp(0, 255)
                .permute(1, 2, 0)
                .numpy()
                .astype(np.uint8)
            )
            image_aug = np.ascontiguousarray(image_aug)

            # Annotate augmented image
            if len(mask_labels_aug) > 0:
                masks_aug = mask_labels_aug.numpy().astype(bool)
                detections_aug = sv.Detections(
                    xyxy=sv.mask_to_xyxy(masks=masks_aug),
                    mask=masks_aug,
                    class_id=class_labels_aug.numpy(),
                )
                annotated_image_aug = mask_annotator.annotate(
                    scene=image_aug.copy(), detections=detections_aug
                )
            else:
                annotated_image_aug = image_aug

            # Concatenate original and augmented side by side
            import cv2

            # Add text labels
            h, w = annotated_image.shape[:2]
            cv2.putText(
                annotated_image,
                "Original",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                annotated_image_aug,
                "Augmented",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )
            annotated_image = np.concatenate(
                [annotated_image, annotated_image_aug], axis=1
            )

        # Save
        output_path = output_dir / f"{CONFIG['split']}_sample_{idx:03d}.jpg"
        Image.fromarray(annotated_image).save(output_path)
        print(f"Saved: {output_path} ({len(class_labels)} instances)")

    print(f"All visualizations saved to {output_dir}")


if __name__ == "__main__":
    visualize()
