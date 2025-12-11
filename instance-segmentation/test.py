from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import supervision as sv
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForUniversalSegmentation

# Configuration
CONFIG = {
    "model_path": "./outputs",
    "dataset_path": "../datasets/yolo/crack",
    "output_dir": "./predictions",
}


class InstanceSegmentationModel(ABC):
    """Base class for instance segmentation models."""

    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

    @abstractmethod
    def predict(self, img: np.ndarray | Image.Image) -> sv.Detections:
        """Predict instance segmentation on an image.

        Args:
            img: Input image as PIL Image or numpy array (RGB)

        Returns:
            supervision Detections with masks, boxes, classes, and confidences
        """
        pass

    def evaluate(self, dataset: sv.DetectionDataset) -> dict:
        """Evaluate model on a detection dataset.

        Args:
            dataset: supervision DetectionDataset

        Returns:
            Dictionary of metrics (mAP, AR, etc.)
        """
        from torchmetrics.detection.mean_ap import MeanAveragePrecision

        metric = MeanAveragePrecision(iou_type="segm", class_metrics=True)
        predictions = []
        targets = []

        for image_path in dataset.image_paths:
            # Load and predict
            image = Image.open(image_path).convert("RGB")
            detections = self.predict(image)

            # Get ground truth annotations
            annotations = dataset.annotations[image_path]

            # Format predictions
            if len(detections) > 0:
                predictions.append(
                    {
                        "masks": torch.from_numpy(detections.mask).to(torch.bool),
                        "labels": torch.from_numpy(detections.class_id),
                        "scores": torch.from_numpy(detections.confidence),
                    }
                )
            else:
                h, w = image.height, image.width
                predictions.append(
                    {
                        "masks": torch.zeros((0, h, w), dtype=torch.bool),
                        "labels": torch.tensor([], dtype=torch.int64),
                        "scores": torch.tensor([], dtype=torch.float32),
                    }
                )

            # Format ground truth
            if annotations.mask is not None and len(annotations.mask) > 0:
                targets.append(
                    {
                        "masks": torch.from_numpy(annotations.mask).to(torch.bool),
                        "labels": torch.from_numpy(annotations.class_id),
                    }
                )
            else:
                h, w = image.height, image.width
                targets.append(
                    {
                        "masks": torch.zeros((0, h, w), dtype=torch.bool),
                        "labels": torch.tensor([], dtype=torch.int64),
                    }
                )

        # Compute metrics
        metric.update(predictions, targets)
        metrics = metric.compute()

        # Convert to regular dict with float values
        return {k: v.item() if torch.is_tensor(v) else v for k, v in metrics.items()}

    @classmethod
    @abstractmethod
    def from_pretrained(cls, model_path: str, **kwargs):
        """Load model from pretrained checkpoint."""
        pass


class Mask2FormerModel(InstanceSegmentationModel):
    """Mask2Former instance segmentation model wrapper."""

    def __init__(
        self,
        model: AutoModelForUniversalSegmentation,
        image_processor: AutoImageProcessor,
        threshold: float = 0.5,
        device: str = "cuda",
    ):
        super().__init__(device)
        self.model = model.to(self.device)
        self.image_processor = image_processor
        self.threshold = threshold
        self.model.eval()

    def predict(self, img: np.ndarray | Image.Image) -> sv.Detections:
        """Predict instance segmentation using Mask2Former."""
        # Convert to PIL if numpy
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)

        # Preprocess
        inputs = self.image_processor(images=img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-process
        target_size = (img.height, img.width)
        results = self.image_processor.post_process_instance_segmentation(
            outputs,
            threshold=self.threshold,
            target_sizes=[target_size],
            return_binary_maps=True,
        )[0]

        # Convert to supervision format
        if results["segments_info"]:
            masks = results["segmentation"].cpu().numpy().astype(bool)
            segments = results["segments_info"]
            return sv.Detections(
                xyxy=sv.mask_to_xyxy(masks),
                mask=masks,
                class_id=np.array([seg["label_id"] for seg in segments]),
                confidence=np.array([seg["score"] for seg in segments]),
            )
        else:
            # Return empty detections
            return sv.Detections.empty()

    @classmethod
    def from_pretrained(
        cls, model_path: str, threshold: float = 0.5, device: str = "cuda"
    ):
        """Load Mask2Former model from pretrained checkpoint.

        Args:
            model_path: Path to model checkpoint
            threshold: Confidence threshold for predictions
            device: Device to load model on

        Returns:
            Mask2FormerModel instance
        """
        model = AutoModelForUniversalSegmentation.from_pretrained(model_path)
        image_processor = AutoImageProcessor.from_pretrained(model_path)
        return cls(model, image_processor, threshold, device)


def test():
    """Test the Mask2Former model on test images."""
    print("Loading model...")
    model = Mask2FormerModel.from_pretrained(
        CONFIG["model_path"], threshold=0.5, device="cuda"
    )
    print(f"Model loaded on {model.device}")

    # Load test dataset for evaluation
    dataset_path = Path(CONFIG["dataset_path"])

    print("\n=== Loading test dataset for evaluation ===")
    test_dataset = sv.DetectionDataset.from_yolo(
        images_directory_path=str(dataset_path / "images" / "test"),
        annotations_directory_path=str(dataset_path / "labels" / "test"),
        data_yaml_path=str(dataset_path / "data.yaml"),
    )
    print(f"Loaded {len(test_dataset)} test samples")

    # Evaluate on test set
    print("\n=== Evaluating model on test set ===")
    metrics = model.evaluate(test_dataset)

    print("\n=== Test Set Metrics ===")
    print(f"mAP: {metrics.get('map', 0.0):.4f}")
    print(f"mAP@50: {metrics.get('map_50', 0.0):.4f}")
    print(f"mAP@75: {metrics.get('map_75', 0.0):.4f}")
    print(f"mAR@100: {metrics.get('mar_100', 0.0):.4f}")

    # Print per-class metrics if available
    for key, value in metrics.items():
        if key.startswith("map_") and key not in ["map", "map_50", "map_75"]:
            print(f"{key}: {value:.4f}")

    # Visualize predictions
    print("\n=== Generating visualizations ===")
    test_images_dir = dataset_path / "images" / "test"
    image_files = sorted(list(test_images_dir.glob("*.jpg")))

    # Create output directory
    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Annotator for visualization
    mask_annotator = sv.MaskAnnotator()

    print(f"Processing {len(image_files)} test images...")

    for img_path in image_files:
        # Load image
        image = Image.open(img_path).convert("RGB")

        # Predict using the model wrapper
        detections = model.predict(image)

        # Annotate image
        if len(detections) > 0:
            annotated_image = mask_annotator.annotate(
                scene=np.array(image), detections=detections
            )
        else:
            annotated_image = np.array(image)

        # Save
        output_path = output_dir / f"{img_path.stem}_pred.jpg"
        Image.fromarray(annotated_image).save(output_path)
        print(f"Saved: {output_path} ({len(detections)} detections)")

    print(f"\nAll predictions saved to {output_dir}")


if __name__ == "__main__":
    test()
