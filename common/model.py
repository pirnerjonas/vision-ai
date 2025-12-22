from pathlib import Path
from typing import Protocol

import numpy as np
import supervision as sv


class SegmentationModel(Protocol):
    """Unified protocol for all segmentation models."""

    name: str

    def labels(self) -> list[str]: ...

    def predict(self, image: np.ndarray) -> sv.Detections: ...


class SmpSemanticSegmentationModel:
    """Semantic segmentation using segmentation-models-pytorch."""

    def __init__(self, model_path: str | Path, device: str = "cpu"):
        import albumentations as A
        import segmentation_models_pytorch as smp

        model = smp.from_pretrained(str(model_path))
        model.to(device)
        model.eval()

        self.model = model
        self.transform = A.Compose.from_pretrained(str(model_path))
        self.device = device
        self.name = "SMP Semantic Segmentation"

    def labels(self) -> list[str]:
        return list(self.model.classes) if self.model.classes else ["foreground"]

    def predict(self, image: np.ndarray) -> sv.Detections:
        import cv2
        import torch

        original_h, original_w = image.shape[:2]

        # Transform and predict
        transformed = self.transform(image=image)
        image_tensor = transformed["image"].unsqueeze(0).to(self.device)

        with torch.inference_mode():
            output = self.model(image_tensor)
            mask = output.squeeze().cpu().numpy()

        # Resize to original size
        mask_resized = cv2.resize(
            mask, (original_w, original_h), interpolation=cv2.INTER_LINEAR
        )

        # Convert to binary mask
        binary_mask = (mask_resized > 0.5).astype(bool)
        masks = binary_mask[np.newaxis, ...]

        xyxy = sv.mask_to_xyxy(masks=masks)

        return sv.Detections(
            xyxy=xyxy,
            mask=masks,
            class_id=np.array([0]),
            confidence=np.array([mask_resized.max()]),
        )


class Mask2FormerInstanceSegmentationModel:
    """Instance segmentation using HuggingFace Mask2Former."""

    def __init__(self, model_path: str | Path, device: str = "cpu"):
        from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

        self.processor = AutoImageProcessor.from_pretrained(str(model_path))
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
            str(model_path)
        )
        self.model.to(device)
        self.model.eval()
        self.device = device
        self.name = "Mask2Former Instance Segmentation"

    def labels(self) -> list[str]:
        return list(self.model.config.id2label.values())

    def predict(self, image: np.ndarray) -> sv.Detections:
        import torch

        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_instance_segmentation(
            outputs, target_sizes=[image.shape[:2]]
        )[0]

        segmentation = results["segmentation"].cpu().numpy()
        segments_info = results["segments_info"]

        masks = []
        class_ids = []
        scores = []

        for segment in segments_info:
            instance_mask = (segmentation == segment["id"]).astype(bool)
            masks.append(instance_mask)
            class_ids.append(segment["label_id"])
            scores.append(segment["score"])

        if len(masks) == 0:
            return sv.Detections.empty()

        masks = np.array(masks)
        xyxy = sv.mask_to_xyxy(masks=masks)

        return sv.Detections(
            xyxy=xyxy,
            mask=masks,
            class_id=np.array(class_ids),
            confidence=np.array(scores),
        )


class UltralyticsInstanceSegmentationModel:
    """Instance segmentation using Ultralytics YOLO."""

    def __init__(self, model_path: str | Path, device: str = "cpu"):
        import ultralytics

        self.model = ultralytics.YOLO(str(model_path))
        self.device = device
        self.name = "Ultralytics YOLO Instance Segmentation"

    def labels(self) -> list[str]:
        return list(self.model.names.values())

    def predict(self, image: np.ndarray) -> sv.Detections:
        results = self.model(image, device=self.device)

        boxes = []
        masks = []
        scores = []
        class_ids = []

        for result in results:
            for box in result.boxes:
                boxes.append(box.xyxy[0].cpu().numpy())
                scores.append(float(box.conf[0].cpu().numpy()))
                class_ids.append(int(box.cls[0].cpu().numpy()))
            if result.masks is not None:
                for mask in result.masks.data:
                    masks.append(mask.cpu().numpy().astype(bool))

        if len(boxes) == 0:
            return sv.Detections.empty()

        return sv.Detections(
            xyxy=np.array(boxes),
            mask=np.array(masks) if masks else None,
            confidence=np.array(scores),
            class_id=np.array(class_ids),
        )


# Utility functions for evaluation and visualization


def evaluate(
    model: SegmentationModel, dataset: sv.DetectionDataset, device: str = "cpu"
) -> dict:
    """Evaluate model on a dataset using IoU metric."""
    import torch
    from torchmetrics.classification import BinaryJaccardIndex

    metric = BinaryJaccardIndex().to(device)

    for _, image, annotations in dataset:
        if annotations.mask is None or len(annotations.mask) == 0:
            continue

        gt_mask = _detections_to_binary_mask(annotations, image.shape[:2])
        pred = model.predict(image)
        pred_mask = _detections_to_binary_mask(pred, image.shape[:2])

        gt_tensor = torch.from_numpy(gt_mask).to(device)
        pred_tensor = torch.from_numpy(pred_mask).to(device)
        metric.update(pred_tensor, gt_tensor)

    return {"iou": metric.compute().item()}


def visualize_predictions(
    model: SegmentationModel,
    dataset: sv.DetectionDataset,
    output_dir: str | Path,
    max_images: int | None = None,
):
    """Save side-by-side comparison of predictions vs ground truth."""
    from PIL import Image, ImageDraw, ImageFont

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX, opacity=0.5)
    label_annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER)

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24
        )
    except OSError:
        font = ImageFont.load_default()

    for idx, (image_path, image, annotations) in enumerate(dataset):
        if max_images and idx >= max_images:
            break

        detections = model.predict(image)

        # Annotate
        pred_annotated = label_annotator.annotate(
            mask_annotator.annotate(image.copy(), detections),
            detections,
            [f"P{i}" for i in range(len(detections))],
        )
        gt_annotated = label_annotator.annotate(
            mask_annotator.annotate(image.copy(), annotations),
            annotations,
            [f"GT{i}" for i in range(len(annotations))],
        )

        # Side-by-side
        h, w = image.shape[:2]
        comparison = Image.fromarray(np.hstack([gt_annotated, pred_annotated]))
        draw = ImageDraw.Draw(comparison)
        draw.text((20, 20), "Ground Truth", fill=(255, 255, 255), font=font)
        draw.text(
            (w + 20, 20),
            f"Predictions ({len(detections)})",
            fill=(255, 255, 255),
            font=font,
        )

        comparison.save(output_dir / f"{Path(image_path).stem}_comparison.jpg")


def _detections_to_binary_mask(detections: sv.Detections, shape: tuple) -> np.ndarray:
    """Convert detections to a single binary mask."""
    import cv2

    if detections.mask is None or len(detections.mask) == 0:
        return np.zeros(shape, dtype=np.uint8)

    binary_mask = np.any(detections.mask, axis=0).astype(np.uint8)

    if binary_mask.shape != shape:
        binary_mask = cv2.resize(
            binary_mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST
        )

    return binary_mask
