import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import supervision as sv

from common.model import (
    Mask2FormerInstanceSegmentationModel,
    evaluate,
    visualize_predictions,
)

# Configuration
CONFIG = {
    "model_path": "./outputs",
    "dataset_path": "../datasets/yolo/zinnperle-segmentation",
    "output_dir": "./predictions",
    "split": "test",
}


def test():
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model
    model_path = Path(CONFIG["model_path"])
    print(f"Loading model from {model_path}...")
    model = Mask2FormerInstanceSegmentationModel(model_path, device=device)
    print(f"✓ Model loaded: {model.name}")

    # Load dataset
    dataset_path = Path(CONFIG["dataset_path"])
    split = CONFIG["split"]

    print(f"Loading {split} dataset from {dataset_path}...")
    dataset = sv.DetectionDataset.from_yolo(
        images_directory_path=str(dataset_path / "images" / split),
        annotations_directory_path=str(dataset_path / "labels" / split),
        data_yaml_path=str(dataset_path / "dataset.yaml"),
        force_masks=True,
    )
    print(f"✓ Loaded {len(dataset)} images")

    # Evaluate
    print(f"\nEvaluating model on {split} set...")
    metrics = evaluate(model, dataset, device=device)

    print(f"\n{'=' * 60}")
    print(f"Test Set Results ({len(dataset)} images):")
    print(f"{'=' * 60}")
    print(f"IoU: {metrics['iou']:.4f}")
    print(f"{'=' * 60}")

    # Visualize
    output_dir = Path(CONFIG["output_dir"])
    if output_dir:
        print("\nGenerating visualizations...")
        visualize_predictions(model, dataset, output_dir)
        print(f"✓ Predictions saved to {output_dir}")


if __name__ == "__main__":
    test()
