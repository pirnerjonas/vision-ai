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
            Dictionary with map_50, mar_50, pixel_iou, pixel_dice,
            pixel_precision, pixel_recall
        """
        from torchmetrics import MetricCollection
        from torchmetrics.classification import (
            BinaryJaccardIndex,
            BinaryPrecision,
            BinaryRecall,
        )
        from torchmetrics.detection.mean_ap import MeanAveragePrecision

        # Initialize metric collections
        instance_metrics = MetricCollection(
            {
                "map_50": MeanAveragePrecision(iou_type="segm", iou_thresholds=[0.5]),
            }
        )
        pixel_metrics = MetricCollection(
            {
                "pixel_iou": BinaryJaccardIndex(),
                "pixel_precision": BinaryPrecision(),
                "pixel_recall": BinaryRecall(),
            }
        )

        # Collect data
        instance_preds, instance_targets = [], []
        pixel_preds, pixel_targets = [], []

        for _, image, annotations in dataset:
            detections = self.predict(image)

            h, w = image.shape[:2]
            num_dets = len(detections)
            num_gts = len(annotations) if annotations.mask is not None else 0

            # Instance data
            instance_preds.append(
                {
                    "masks": torch.from_numpy(detections.mask).to(torch.bool)
                    if num_dets > 0
                    else torch.zeros((0, h, w), dtype=torch.bool),
                    "labels": torch.from_numpy(detections.class_id)
                    if num_dets > 0
                    else torch.tensor([], dtype=torch.int64),
                    "scores": torch.from_numpy(detections.confidence)
                    if num_dets > 0
                    else torch.tensor([], dtype=torch.float32),
                }
            )
            instance_targets.append(
                {
                    "masks": torch.from_numpy(annotations.mask).to(torch.bool)
                    if num_gts > 0
                    else torch.zeros((0, h, w), dtype=torch.bool),
                    "labels": torch.from_numpy(annotations.class_id)
                    if num_gts > 0
                    else torch.tensor([], dtype=torch.int64),
                }
            )

            # Pixel data (merge all instances)
            pred_mask = (
                detections.mask.any(axis=0) if num_dets > 0 else np.zeros((h, w))
            )
            gt_mask = annotations.mask.any(axis=0) if num_gts > 0 else np.zeros((h, w))
            pixel_preds.append(torch.from_numpy(pred_mask.astype(np.uint8)))
            pixel_targets.append(torch.from_numpy(gt_mask.astype(np.uint8)))

        # Compute metrics
        instance_metrics.update(instance_preds, instance_targets)
        pixel_metrics.update(
            torch.stack(pixel_preds).flatten(), torch.stack(pixel_targets).flatten()
        )

        results_instance = instance_metrics.compute()
        results_pixel = pixel_metrics.compute()

        # MetricCollection returns flattened dict
        metrics_dict = {
            "map_50": results_instance["map"].item(),
            "mar_50": results_instance["mar_100"].item(),
            **{k: v.item() for k, v in results_pixel.items()},
        }

        # Add Dice coefficient
        p, r = metrics_dict["pixel_precision"], metrics_dict["pixel_recall"]
        metrics_dict["pixel_dice"] = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

        return metrics_dict

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
    print("\n📊 Instance Metrics (IoU@50):")
    print(f"  mAP@50: {metrics['map_50']:.4f}")
    print(f"  mAR@50: {metrics['mar_50']:.4f}")

    print("\n🎨 Pixel Metrics:")
    print(f"  IoU:       {metrics['pixel_iou']:.4f}")
    print(f"  Dice:      {metrics['pixel_dice']:.4f}")
    print(f"  Precision: {metrics['pixel_precision']:.2%}")
    print(f"  Recall:    {metrics['pixel_recall']:.2%}")

    # Visualize predictions
    print("\n=== Generating visualizations ===")

    # Create output directory
    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Annotators for visualization with different colors per instance
    # ColorPalette.DEFAULT cycles through different colors automatically
    mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX, opacity=0.5)
    label_annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER)

    print(f"Processing {len(test_dataset)} test images...")

    for image_path, image, gt_annotations in test_dataset:
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
            scene=image.copy(), detections=gt_annotations
        )
        gt_annotated = label_annotator.annotate(
            scene=gt_annotated,
            detections=gt_annotations,
            labels=[f"GT{i}" for i in range(len(gt_annotations))],
        )

        # Create side-by-side comparison
        # Add text labels
        pred_labeled = pred_annotated.copy()
        gt_labeled = gt_annotated.copy()

        # Combine horizontally
        comparison = np.hstack([gt_labeled, pred_labeled])

        # Add text headers (optional - using PIL for better text)
        from PIL import ImageDraw, ImageFont

        comparison_pil = Image.fromarray(comparison)
        draw = ImageDraw.Draw(comparison_pil)

        # Try to use a font, fallback to default
        try:
            font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
            font = ImageFont.truetype(font_path, 24)
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

        # Save comparison
        output_path = output_dir / f"{Path(image_path).stem}_comparison.jpg"
        comparison_pil.save(output_path)

        gt_count = len(gt_annotations) if gt_annotations.mask is not None else 0
        print(f"Saved: {output_path.name} (GT: {gt_count}, Pred: {len(detections)})")

    print(f"\nAll predictions saved to {output_dir}")


if __name__ == "__main__":
    test()
