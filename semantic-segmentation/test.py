import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import cv2
import numpy as np
import supervision as sv

from common.model import VisionModel

# Configuration
# Note: Update these paths to match your training configuration
CONFIG = {
    "model_path": "./outputs/crack-segmentation",  # Path to saved model directory
    "dataset_path": "../datasets/yolo/crack",
    "output_dir": "./predictions",
    "threshold": 0.5,
    "split": "test",
}


def test():
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model using VisionModel
    model_path = Path(CONFIG["model_path"])
    print(f"Loading model from {model_path}...")
    model = VisionModel.from_smp(model_path, model_type="semantic", device=device)
    print("✓ Model loaded successfully")

    # Load dataset
    dataset_path = Path(CONFIG["dataset_path"])
    split = CONFIG["split"]

    print(f"Loading {split} dataset from {dataset_path}...")
    dataset = sv.DetectionDataset.from_yolo(
        images_directory_path=str(dataset_path / "images" / split),
        annotations_directory_path=str(dataset_path / "labels" / split),
        data_yaml_path=str(dataset_path / "data.yaml"),
        force_masks=True,
    )
    print(f"✓ Loaded {len(dataset)} images")

    # Evaluate model using high-level function
    print(f"\nEvaluating model on {split} set...")
    metrics = model.evaluate(dataset)

    # Print results
    print(f"\n{'=' * 60}")
    print(f"Test Set Results ({len(dataset)} images):")
    print(f"{'=' * 60}")
    print(f"IoU: {metrics['iou']:.4f}")
    print(f"{'=' * 60}")

    # Optional: Generate predictions for visualization
    output_dir = Path(CONFIG["output_dir"])
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        print("\nGenerating visualizations...")

        for image_path, image, annotations in dataset:
            # Get prediction
            detections = model.predict(image)

            # Merge instance masks to semantic mask
            if len(detections) > 0 and detections.mask is not None:
                pred_mask = np.any(detections.mask, axis=0).astype(np.uint8)
            else:
                pred_mask = np.zeros(image.shape[:2], dtype=np.uint8)

            # Resize prediction to match original image size
            if pred_mask.shape != image.shape[:2]:
                pred_mask = cv2.resize(
                    pred_mask,
                    (image.shape[1], image.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )

            # Get ground truth mask
            if annotations.mask is not None and len(annotations.mask) > 0:
                gt_mask = np.any(annotations.mask, axis=0).astype(np.uint8)
            else:
                gt_mask = np.zeros(image.shape[:2], dtype=np.uint8)

            # Create visualization
            overlay = image.copy()
            overlay[pred_mask == 1] = [0, 255, 0]  # Green for prediction
            overlay[gt_mask == 1] = [255, 0, 0]  # Red for ground truth
            overlay[(pred_mask == 1) & (gt_mask == 1)] = [255, 255, 0]  # Yellow overlap

            result = cv2.addWeighted(image, 0.6, overlay, 0.4, 0)

            # Add contours
            contours, _ = cv2.findContours(
                pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(result, contours, -1, (0, 255, 0), 2)

            # Save
            img_name = Path(image_path).stem
            output_path = output_dir / f"{img_name}_pred.png"
            result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(output_path), result_bgr)

        print(f"✓ Predictions saved to {output_dir}")


if __name__ == "__main__":
    test()
