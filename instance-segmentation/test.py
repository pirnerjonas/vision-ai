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
            Dictionary of metrics (mAP, AR, precision, recall, F1, etc.)
        """
        from torchmetrics.detection.mean_ap import MeanAveragePrecision

        # Compute metrics at different IoU thresholds
        metric_iou50 = MeanAveragePrecision(
            iou_type="segm", iou_thresholds=[0.5], class_metrics=True
        )
        metric_all = MeanAveragePrecision(iou_type="segm", class_metrics=True)
        predictions = []
        targets = []

        # Additional metrics for better understanding
        total_pred_masks = 0
        total_gt_masks = 0
        matched_at_50 = 0  # Count matches at IoU ≥ 0.5

        # Pixel-level metrics (ignore instance boundaries)
        all_pred_pixels = []
        all_gt_pixels = []

        for image_path in dataset.image_paths:
            # Load and predict
            image = Image.open(image_path).convert("RGB")
            detections = self.predict(image)

            # Get ground truth annotations
            annotations = dataset.annotations[image_path]

            # Count for additional metrics
            total_pred_masks += len(detections) if len(detections) > 0 else 0
            gt_count = len(annotations.mask) if annotations.mask is not None else 0
            total_gt_masks += gt_count

            # Calculate IoU for matches (simplified)
            if len(detections) > 0 and gt_count > 0:
                # Count as matched if any detection exists (rough estimate)
                matched_at_50 += min(len(detections), gt_count)

            # Accumulate pixel-level masks (merge all instances into single mask)
            h, w = image.height, image.width

            # Merge prediction masks into single binary mask
            if len(detections) > 0:
                pred_pixel_mask = detections.mask.any(axis=0).astype(np.uint8)
            else:
                pred_pixel_mask = np.zeros((h, w), dtype=np.uint8)
            all_pred_pixels.append(pred_pixel_mask)

            # Merge GT masks into single binary mask
            if gt_count > 0:
                gt_pixel_mask = annotations.mask.any(axis=0).astype(np.uint8)
            else:
                gt_pixel_mask = np.zeros((h, w), dtype=np.uint8)
            all_gt_pixels.append(gt_pixel_mask)

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

        # Compute metrics at both IoU thresholds
        metric_iou50.update(predictions, targets)
        metric_all.update(predictions, targets)

        metrics_iou50 = metric_iou50.compute()
        metrics_all = metric_all.compute()

        # Combine metrics - use IoU@50 as primary, add overall metrics
        metrics_dict = {
            k: v.item() if torch.is_tensor(v) else v for k, v in metrics_all.items()
        }

        # Add IoU@50 specific metrics (most important for deployment)
        metrics_dict["map_50"] = metrics_iou50["map"].item()
        metrics_dict["mar_50"] = metrics_iou50["mar_100"].item()  # Recall @ IoU 50

        # Add count-based metrics
        metrics_dict["total_predictions"] = total_pred_masks
        metrics_dict["total_ground_truth"] = total_gt_masks
        metrics_dict["detection_rate"] = (
            matched_at_50 / total_gt_masks if total_gt_masks > 0 else 0.0
        )
        metrics_dict["precision_rough"] = (
            matched_at_50 / total_pred_masks if total_pred_masks > 0 else 0.0
        )

        # Compute pixel-level IoU (ignores instance boundaries)
        if all_pred_pixels and all_gt_pixels:
            # Stack all images and compute union/intersection across entire dataset
            all_pred = np.stack(all_pred_pixels)
            all_gt = np.stack(all_gt_pixels)

            intersection = (all_pred & all_gt).sum()
            union = (all_pred | all_gt).sum()

            pixel_iou = intersection / union if union > 0 else 0.0
            pixel_precision = (
                intersection / all_pred.sum() if all_pred.sum() > 0 else 0.0
            )
            pixel_recall = intersection / all_gt.sum() if all_gt.sum() > 0 else 0.0
            pixel_dice = (
                2 * intersection / (all_pred.sum() + all_gt.sum())
                if (all_pred.sum() + all_gt.sum()) > 0
                else 0.0
            )

            metrics_dict["pixel_iou"] = pixel_iou
            metrics_dict["pixel_precision"] = pixel_precision
            metrics_dict["pixel_recall"] = pixel_recall
            metrics_dict["pixel_dice"] = pixel_dice

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
    print("\n📊 COCO-style Metrics (IoU-based):")
    print(f"  mAP (IoU 0.50:0.95): {metrics.get('map', 0.0):.4f}")
    print(f"  mAP@50 (IoU ≥ 0.50): {metrics.get('map_50', 0.0):.4f} ← Precision @ 50%")
    print(f"  mAR@50 (IoU ≥ 0.50): {metrics.get('mar_50', 0.0):.4f} ← Recall @ 50%")
    print(f"  mAP@75 (IoU ≥ 0.75): {metrics.get('map_75', 0.0):.4f}")
    print(f"  mAR@100 (all IoUs):  {metrics.get('mar_100', 0.0):.4f}")

    print("\n🔢 Detection Counts:")
    print(f"  Total predictions:   {metrics.get('total_predictions', 0)}")
    print(f"  Total ground truth:  {metrics.get('total_ground_truth', 0)}")
    print(f"  Detection rate:      {metrics.get('detection_rate', 0.0):.2%}")
    print(f"  Rough precision:     {metrics.get('precision_rough', 0.0):.2%}")

    print("\n🎨 Pixel-Level Metrics (ignores instance splits):")
    print(f"  Pixel IoU:           {metrics.get('pixel_iou', 0.0):.4f}")
    print(f"  Pixel Dice:          {metrics.get('pixel_dice', 0.0):.4f}")
    print(f"  Pixel Precision:     {metrics.get('pixel_precision', 0.0):.2%}")
    print(f"  Pixel Recall:        {metrics.get('pixel_recall', 0.0):.2%}")

    print("\n💡 Interpretation (for IoU@50 deployment):")
    map_50 = metrics.get("map_50", 0.0)
    mar_50 = metrics.get("mar_50", 0.0)
    pixel_iou = metrics.get("pixel_iou", 0.0)

    print(f"  Instance Precision (mAP@50): {map_50:.1%} of predictions are correct")
    print(f"  Instance Recall (mAR@50):    {mar_50:.1%} of cracks are found")
    print(f"  Pixel IoU:                   {pixel_iou:.1%} crack coverage quality")

    if pixel_iou >= 0.7:
        print("  ✅ Excellent pixel-level coverage!")
    elif pixel_iou >= 0.5:
        print("  ✅ Good pixel-level coverage")
    elif pixel_iou >= 0.3:
        print("  ⚠️  Moderate pixel coverage")
    else:
        print("  ❌ Poor pixel coverage")

    if mar_50 >= 0.9:
        print("  ✅ Excellent recall - catching almost all cracks!")
    elif mar_50 >= 0.7:
        print("  ✅ Good recall - finding most cracks")
    elif mar_50 >= 0.5:
        print("  ⚠️  Moderate recall - missing some cracks")
    else:
        print("  ❌ Low recall - missing many cracks")

    # Note about pixel vs instance metrics
    if pixel_iou > map_50 * 1.5:
        print("\n  📝 Note: Pixel IoU is much higher than instance mAP@50.")
        print("     This suggests your model covers cracks well but may")
        print("     split/merge instances differently than ground truth.")

    # Print per-class metrics if available
    print("\n📋 Per-class metrics:")
    for key, value in metrics.items():
        if key.startswith("map_") and key not in [
            "map",
            "map_50",
            "map_75",
            "map_small",
            "map_medium",
            "map_large",
        ]:
            print(f"  {key}: {value:.4f}")

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

    for image_path in test_dataset.image_paths:
        # Load image
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)

        # Get ground truth annotations
        gt_annotations = test_dataset.annotations[image_path]

        # Get predictions
        detections = model.predict(image)

        # Annotate predictions (each instance gets different color)
        pred_annotated = mask_annotator.annotate(
            scene=image_np.copy(), detections=detections
        )
        pred_annotated = label_annotator.annotate(
            scene=pred_annotated,
            detections=detections,
            labels=[f"P{i}" for i in range(len(detections))],
        )

        # Annotate ground truth (each instance gets different color)
        gt_annotated = mask_annotator.annotate(
            scene=image_np.copy(), detections=gt_annotations
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
        h, w = image_np.shape[:2]
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
