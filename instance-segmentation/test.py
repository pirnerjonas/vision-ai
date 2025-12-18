import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import supervision as sv
from PIL import Image, ImageDraw, ImageFont

from common.model import VisionModel

# Configuration
CONFIG = {
    "model_path": "./outputs",
    "dataset_path": "../datasets/yolo/crack",
    "output_dir": "./predictions",
    "split": "test",
}


def test():
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model using VisionModel
    model_path = Path(CONFIG["model_path"])
    print(f"Loading model from {model_path}...")
    model = VisionModel.from_hf(model_path, model_type="instance", device=device)
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

        # Annotators for visualization
        mask_annotator = sv.MaskAnnotator(
            color_lookup=sv.ColorLookup.INDEX, opacity=0.5
        )
        label_annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER)

        for image_path, image, annotations in dataset:
            # Get predictions
            detections = model.predict(image)

            # Annotate predictions (each instance gets different color)
            pred_annotated = mask_annotator.annotate(
                scene=image.copy(), detections=detections
            )
            pred_annotated = label_annotator.annotate(
                scene=pred_annotated,
                detections=detections,
                labels=[f"P{i}" for i in range(len(detections))],
            )

            # Annotate ground truth (each instance gets different color)
            gt_annotated = mask_annotator.annotate(
                scene=image.copy(), detections=annotations
            )
            gt_annotated = label_annotator.annotate(
                scene=gt_annotated,
                detections=annotations,
                labels=[f"GT{i}" for i in range(len(annotations))],
            )

            # Create side-by-side comparison
            comparison = np.hstack([gt_annotated, pred_annotated])
            comparison_pil = Image.fromarray(comparison)
            draw = ImageDraw.Draw(comparison_pil)

            # Try to use a font, fallback to default
            try:
                font = ImageFont.truetype(
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24
                )
            except Exception:
                font = ImageFont.load_default()

            # Add labels
            h, w = image.shape[:2]
            draw.text((20, 20), "Ground Truth", fill=(255, 255, 255), font=font)
            draw.text(
                (w + 20, 20),
                f"Predictions ({len(detections)})",
                fill=(255, 255, 255),
                font=font,
            )

            # Save
            img_name = Path(image_path).stem
            output_path = output_dir / f"{img_name}_comparison.jpg"
            comparison_pil.save(output_path)

        print(f"✓ Predictions saved to {output_dir}")


if __name__ == "__main__":
    test()
