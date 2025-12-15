from pathlib import Path

import cv2
import numpy as np
from dataset import YOLOSemanticDataset, get_visualization_transforms

# Configuration
CONFIG = {
    "dataset_path": "../datasets/yolo/zinnperle-segmentation",
    "split": "train",  # Can be "train", "valid", or "test"
    "output_dir": "./visualizations",
    "num_samples": 10,  # Number of samples to visualize
    "show_augmentations": True,  # Show original vs augmented side by side
    "image_size": 512,
}


def overlay_mask_on_image(image, mask, color=(0, 255, 0), alpha=0.4):
    """Overlay binary mask on image."""
    overlay = image.copy()
    overlay[mask > 0] = color
    result = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)

    # Add contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, color, 2)

    return result


def visualize():
    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load datasets (original and augmented if requested)
    transform_original = get_visualization_transforms(
        augment=False, image_size=CONFIG["image_size"]
    )
    dataset = YOLOSemanticDataset(
        CONFIG["dataset_path"],
        split=CONFIG["split"],
        transform=transform_original,
    )

    dataset_aug = None
    if CONFIG["show_augmentations"]:
        transform_augmented = get_visualization_transforms(
            augment=True, image_size=CONFIG["image_size"]
        )
        dataset_aug = YOLOSemanticDataset(
            CONFIG["dataset_path"],
            split=CONFIG["split"],
            transform=transform_augmented,
        )

    print(
        f"Visualizing {min(CONFIG['num_samples'], len(dataset))} samples "
        f"from {CONFIG['split']} set..."
    )

    for idx in range(min(CONFIG["num_samples"], len(dataset))):
        # Get original sample
        image, mask = dataset[idx]

        # Convert from tensor to numpy
        if hasattr(image, "numpy"):
            image_np = image.permute(1, 2, 0).numpy()
            image_np = (image_np * 255).astype(np.uint8)
        else:
            image_np = image

        if hasattr(mask, "numpy"):
            mask_np = mask.squeeze().numpy().astype(np.uint8)
        else:
            mask_np = mask.astype(np.uint8)

        # Create visualization
        has_mask = mask_np.sum() > 0
        print(
            f"  Sample {idx}: {'Has mask' if has_mask else 'No mask (background only)'}"
        )

        annotated_image = overlay_mask_on_image(image_np, mask_np)

        # If showing augmentations, get augmented version and concatenate
        if CONFIG["show_augmentations"] and dataset_aug is not None:
            image_aug, mask_aug = dataset_aug[idx]

            # Convert from tensor to numpy
            if hasattr(image_aug, "numpy"):
                image_aug_np = image_aug.permute(1, 2, 0).numpy()
                image_aug_np = (image_aug_np * 255).astype(np.uint8)
            else:
                image_aug_np = image_aug

            if hasattr(mask_aug, "numpy"):
                mask_aug_np = mask_aug.squeeze().numpy().astype(np.uint8)
            else:
                mask_aug_np = mask_aug.astype(np.uint8)

            # Annotate augmented image
            annotated_image_aug = overlay_mask_on_image(image_aug_np, mask_aug_np)

            # Add text labels
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

            # Concatenate original and augmented side by side
            annotated_image = np.concatenate(
                [annotated_image, annotated_image_aug], axis=1
            )

        # Save
        output_path = output_dir / f"{CONFIG['split']}_sample_{idx:03d}.png"
        cv2.imwrite(str(output_path), cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
        print(f"Saved: {output_path}")

    print(f"\nAll visualizations saved to {output_dir}")


if __name__ == "__main__":
    visualize()
