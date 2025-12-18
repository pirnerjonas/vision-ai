import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import supervision as sv

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
        print("\nGenerating visualizations...")
        model.visualize_predictions(dataset, output_dir)
        print(f"✓ Predictions saved to {output_dir}")


if __name__ == "__main__":
    test()
