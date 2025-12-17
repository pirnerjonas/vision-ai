import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import cv2
import numpy as np

from common.model import SemanticSegmentationModel

# Configuration
# Note: Update these paths to match your training configuration
CONFIG = {
    "model_path": "./outputs/crack-segmentation",  # Path to saved model directory
    "dataset_path": "../datasets/yolo/crack",
    "output_dir": "./predictions",
    "threshold": 0.5,
    "split": "test",
}


def load_ground_truth_mask(label_path, img_shape):
    """Load ground truth mask from YOLO format."""
    mask = np.zeros(img_shape[:2], dtype=np.uint8)

    if not label_path.exists():
        return mask

    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            # Skip class_id, parse polygon coordinates
            coords = np.array(parts[1:], dtype=np.float32)
            coords = coords.reshape(-1, 2)

            # Convert normalized coordinates to absolute
            coords[:, 0] *= img_shape[1]  # x * width
            coords[:, 1] *= img_shape[0]  # y * height

            # Draw filled polygon
            cv2.fillPoly(mask, [coords.astype(np.int32)], 1)

    return mask


def calculate_metrics(pred_mask, gt_mask):
    """Calculate segmentation metrics."""
    pred = pred_mask.astype(np.float32)
    gt = gt_mask.astype(np.float32)

    intersection = (pred * gt).sum()
    union = pred.sum() + gt.sum() - intersection

    iou = (intersection + 1e-6) / (union + 1e-6)
    dice = (2 * intersection + 1e-6) / (pred.sum() + gt.sum() + 1e-6)

    true_positive = intersection
    predicted_positive = pred.sum()
    actual_positive = gt.sum()

    precision = (true_positive + 1e-6) / (predicted_positive + 1e-6)
    recall = (true_positive + 1e-6) / (actual_positive + 1e-6)

    return {"iou": iou, "dice": dice, "precision": precision, "recall": recall}


def test():
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model using SemanticSegmentationModel
    model_path = Path(CONFIG["model_path"])
    print(f"Loading model from {model_path}...")
    model = SemanticSegmentationModel.from_smp(model_path, device=device)
    print("✓ Model loaded successfully")

    # Get test images
    images_dir = Path(CONFIG["dataset_path"]) / "images" / CONFIG["split"]
    labels_dir = Path(CONFIG["dataset_path"]) / "labels" / CONFIG["split"]

    # Support multiple image extensions
    image_files = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
        image_files.extend(images_dir.glob(ext))
    image_files = sorted(image_files)

    # Create output directory
    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing {len(image_files)} test images...")

    # Metrics accumulator
    all_metrics = []

    for img_path in image_files:
        # Load image
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_image = image.copy()
        original_size = image.shape[:2]

        # Load ground truth mask
        label_path = labels_dir / (img_path.stem + ".txt")
        gt_mask = load_ground_truth_mask(label_path, image.shape)

        # Predict using model
        detections = model.predict(image)

        # Merge all instance masks back into a single semantic mask
        if len(detections) > 0 and detections.mask is not None:
            binary_mask = np.any(detections.mask, axis=0).astype(np.uint8)
        else:
            binary_mask = np.zeros(image.shape[:2], dtype=np.uint8)

        # Resize to original size
        binary_mask = cv2.resize(
            binary_mask,
            (original_size[1], original_size[0]),
            interpolation=cv2.INTER_NEAREST,
        )

        # Calculate metrics
        metrics = calculate_metrics(binary_mask, gt_mask)
        all_metrics.append(metrics)

        # Create left image (original)
        left_image = original_image.copy()
        cv2.putText(
            left_image,
            "Original",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )

        # Create right image (visualization with predictions)
        overlay = original_image.copy()
        overlay[binary_mask == 1] = [0, 255, 0]  # Green overlay for prediction
        overlay[gt_mask == 1] = [255, 0, 0]  # Red overlay for ground truth
        overlay[(binary_mask == 1) & (gt_mask == 1)] = [
            255,
            255,
            0,
        ]  # Yellow for overlap
        right_image = cv2.addWeighted(original_image, 0.6, overlay, 0.4, 0)

        # Add mask contours
        contours, _ = cv2.findContours(
            binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(right_image, contours, -1, (0, 255, 0), 2)

        # Add text with metrics
        cv2.putText(
            right_image,
            "Prediction",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )
        text = f"IoU: {metrics['iou']:.3f} | Dice: {metrics['dice']:.3f}"
        cv2.putText(
            right_image,
            text,
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        # Concatenate images side by side
        result = np.concatenate([left_image, right_image], axis=1)

        # Save
        output_path = output_dir / f"{img_path.stem}_pred.png"
        result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_path), result_bgr)
        print(f"Saved: {output_path} | IoU: {metrics['iou']:.3f}")

    # Calculate and print average metrics
    avg_metrics = {
        "iou": np.mean([m["iou"] for m in all_metrics]),
        "dice": np.mean([m["dice"] for m in all_metrics]),
        "precision": np.mean([m["precision"] for m in all_metrics]),
        "recall": np.mean([m["recall"] for m in all_metrics]),
    }

    print(f"\n{'=' * 60}")
    print(f"Test Set Results ({len(image_files)} images):")
    print(f"{'=' * 60}")
    print(f"Average IoU:       {avg_metrics['iou']:.4f}")
    print(f"Average Dice:      {avg_metrics['dice']:.4f}")
    print(f"Average Precision: {avg_metrics['precision']:.4f}")
    print(f"Average Recall:    {avg_metrics['recall']:.4f}")
    print(f"{'=' * 60}")
    print(f"Predictions saved to {output_dir}")


if __name__ == "__main__":
    test()
