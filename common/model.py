from enum import Enum
from pathlib import Path
from typing import Literal

import numpy as np
import supervision as sv


class VisionModelType(Enum):
    SEMANTIC = "semantic"
    INSTANCE = "instance"


def binary_mask_to_detections(binary_mask) -> sv.Detections:
    """Convert a binary semantic segmentation mask to instance detections.

    Uses connected components to separate individual instances and creates
    bounding boxes for each instance.

    Args:
        binary_mask: Binary mask (H, W) with values 0 or 1

    Returns:
        sv.Detections object with instance masks, bounding boxes, and class IDs
    """
    import cv2
    import numpy as np
    from supervision.detection.utils.converters import mask_to_xyxy

    # Find connected components to convert semantic mask to instance masks
    num_labels, labels = cv2.connectedComponents(binary_mask)

    # If no instances found, return empty detections
    if num_labels <= 1:  # Only background
        return sv.Detections.empty()

    # Create instance masks for each connected component
    masks = np.array(
        [(labels == instance_id) for instance_id in range(1, num_labels)], dtype=bool
    )

    # Use supervision's mask_to_xyxy to get bounding boxes
    xyxy = mask_to_xyxy(masks)

    # All instances belong to class 0
    class_ids = np.zeros(len(masks), dtype=int)

    detections = sv.Detections(
        xyxy=xyxy.astype(np.float32),
        mask=masks,
        class_id=class_ids,
    )

    return detections


class VisionModel:
    def __init__(self, model_type: VisionModelType, device: str = "cuda"):
        self.model = None
        self.model_type = model_type
        self.device = device
        # Backend-specific attributes
        self.transform = None  # For SMP models
        self.image_processor = None  # For HF models

    def predict(self, image: np.ndarray) -> sv.Detections:
        """Unified prediction interface that returns sv.Detections.

        Automatically converts model outputs to sv.Detections based on model_type.
        """
        # Get raw prediction based on loaded backend
        if hasattr(self, "_backend") and self._backend == "smp":
            raw_output = self._smp_predict_impl(image)
        elif hasattr(self, "_backend") and self._backend == "hf":
            raw_output = self._hf_predict_impl(image)
        else:
            raise RuntimeError("No model backend loaded")

        # Convert to detections based on model type
        if self.model_type == VisionModelType.SEMANTIC:
            return self._semantic_to_detections(raw_output, image.shape[:2])
        else:  # INSTANCE
            return self._instance_to_detections(raw_output, image.shape[:2])

    def _semantic_to_detections(
        self, mask: np.ndarray, original_shape: tuple
    ) -> sv.Detections:
        """Convert semantic mask to instance detections via connected components."""
        import cv2

        # Resize mask to original image size
        if mask.shape != original_shape:
            mask = cv2.resize(
                mask.astype(np.uint8),
                (original_shape[1], original_shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        return binary_mask_to_detections(mask)

    def _instance_to_detections(
        self, outputs: dict | sv.Detections, original_shape: tuple
    ) -> sv.Detections:
        """Convert instance predictions to sv.Detections."""
        import cv2
        from supervision.detection.utils.converters import mask_to_xyxy

        # If already sv.Detections, return as is (from SMP)
        if isinstance(outputs, sv.Detections):
            return outputs

        # Handle HF dict format
        if isinstance(outputs, dict):
            masks = outputs["masks"]  # [N, H, W]
            scores = outputs.get("scores", np.ones(len(masks)))
            labels = outputs.get("labels", np.zeros(len(masks), dtype=int))

            if len(masks) == 0:
                return sv.Detections.empty()

            # Resize masks to original size if needed
            if masks.shape[1:] != original_shape:
                resized_masks = []
                for mask in masks:
                    resized = cv2.resize(
                        mask.astype(np.uint8),
                        (original_shape[1], original_shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(bool)
                    resized_masks.append(resized)
                masks = np.array(resized_masks)

            # Compute bboxes from masks
            xyxy = mask_to_xyxy(masks)

            return sv.Detections(
                xyxy=xyxy.astype(np.float32),
                mask=masks,
                class_id=labels.astype(int),
                confidence=scores.astype(np.float32),
            )

        raise ValueError(f"Unsupported output format: {type(outputs)}")

    def evaluate(self, dataset: sv.DetectionDataset) -> dict:
        """Evaluate model on a dataset using torchmetrics IoU.

        Works for both semantic and instance segmentation models by converting
        predictions to binary masks for comparison.

        Args:
            dataset: supervision DetectionDataset

        Returns:
            Dictionary with IoU metrics
        """
        import torch
        from torchmetrics.classification import BinaryJaccardIndex

        metric = BinaryJaccardIndex().to(self.device)

        for _, image, annotations in dataset:
            # Skip if no ground truth
            if annotations.mask is None or len(annotations.mask) == 0:
                continue

            # Get ground truth binary mask
            gt_mask = self._detections_to_binary_mask(annotations, image.shape[:2])

            # Get prediction
            pred_detections = self.predict(image)
            pred_mask = self._detections_to_binary_mask(
                pred_detections, image.shape[:2]
            )

            # Convert to torch tensors
            gt_tensor = torch.from_numpy(gt_mask).to(self.device)
            pred_tensor = torch.from_numpy(pred_mask).to(self.device)

            # Update metric
            metric.update(pred_tensor, gt_tensor)

        # Compute final IoU
        iou = metric.compute().item()

        return {"iou": iou}

    @staticmethod
    def _detections_to_binary_mask(
        detections: sv.Detections, shape: tuple
    ) -> np.ndarray:
        """Convert sv.Detections to binary semantic mask.

        Args:
            detections: Detections object with masks
            shape: Target shape (H, W)

        Returns:
            Binary mask as uint8 numpy array
        """
        if detections.mask is None or len(detections.mask) == 0:
            return np.zeros(shape, dtype=np.uint8)

        # Stack all instance masks into single binary mask
        binary_mask = np.any(detections.mask, axis=0).astype(np.uint8)

        # Resize if needed
        if binary_mask.shape != shape:
            import cv2

            binary_mask = cv2.resize(
                binary_mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST
            )

        return binary_mask

    def visualize_predictions(
        self, dataset, output_dir: str | Path, max_images: int | None = None
    ):
        """Visualize predictions for dataset images."""
        import supervision as sv
        from PIL import Image, ImageDraw, ImageFont

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        mask_annotator = sv.MaskAnnotator(
            color_lookup=sv.ColorLookup.INDEX, opacity=0.5
        )
        label_annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER)
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24
        )

        for idx, (image_path, image, annotations) in enumerate(dataset):
            if max_images and idx >= max_images:
                break

            detections = self.predict(image)

            # Annotate predictions and ground truth
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

            # Create side-by-side comparison with labels
            comparison_pil = Image.fromarray(np.hstack([gt_annotated, pred_annotated]))
            draw = ImageDraw.Draw(comparison_pil)
            h, w = image.shape[:2]
            draw.text((20, 20), "Ground Truth", fill=(255, 255, 255), font=font)
            draw.text(
                (w + 20, 20),
                f"Predictions ({len(detections)})",
                fill=(255, 255, 255),
                font=font,
            )

            output_path = output_dir / f"{Path(image_path).stem}_comparison.jpg"
            comparison_pil.save(output_path)

    def _smp_predict_impl(self, image: np.ndarray) -> np.ndarray:
        """SMP prediction implementation - returns semantic binary mask."""
        import torch

        # Apply transforms
        transformed = self.transform(image=image)
        image_tensor = transformed["image"].unsqueeze(0).to(self.device)

        # Make prediction
        with torch.inference_mode():
            output = self.model(image_tensor)
            mask = output.squeeze().cpu().numpy()

        # Threshold mask to binary
        binary_mask = (mask > 0.5).astype(np.uint8)

        return binary_mask

    def _hf_predict_impl(self, image: np.ndarray) -> dict:
        """HF prediction implementation - returns dict with masks, scores, labels."""
        import torch

        # Process image
        inputs = self.image_processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-process to get instance segmentation
        results = self.image_processor.post_process_instance_segmentation(
            outputs, target_sizes=[image.shape[:2]]
        )[0]

        # Extract and convert to numpy
        segmentation = results["segmentation"].cpu().numpy()

        # Convert segmentation map to individual masks
        unique_ids = np.unique(segmentation)
        unique_ids = unique_ids[unique_ids > 0]  # Remove background

        if len(unique_ids) == 0:
            return {
                "masks": np.array([]),
                "scores": np.array([]),
                "labels": np.array([]),
            }

        masks = np.array([segmentation == uid for uid in unique_ids])

        # Extract scores and labels if available
        scores = (
            results.get("scores", np.ones(len(masks))).cpu().numpy()
            if hasattr(results.get("scores", None), "cpu")
            else np.ones(len(masks))
        )
        labels = (
            results.get("labels", np.zeros(len(masks))).cpu().numpy()
            if hasattr(results.get("labels", None), "cpu")
            else np.zeros(len(masks))
        )

        return {"masks": masks, "scores": scores, "labels": labels}

    @classmethod
    def from_smp(
        cls,
        model_path,
        model_type: Literal["semantic", "instance"] = "semantic",
        device: str = "cuda",
    ):
        """Load a segmentation-models-pytorch model.

        Args:
            model_path: Path to saved model directory
            model_type: Type of segmentation ("semantic" or "instance")
            device: Device to load model on

        Returns:
            VisionModel instance
        """
        import albumentations as A
        import segmentation_models_pytorch as smp

        model = smp.from_pretrained(str(model_path))
        model.to(device)
        model.eval()

        # Load transforms from saved model
        transform = A.Compose.from_pretrained(str(model_path))

        # Convert string to enum
        type_enum = (
            VisionModelType.SEMANTIC
            if model_type == "semantic"
            else VisionModelType.INSTANCE
        )

        instance = cls(type_enum, device=device)
        instance.model = model
        instance.transform = transform
        instance._backend = "smp"
        return instance

    @classmethod
    def from_hf(
        cls,
        model_path,
        model_type: Literal["semantic", "instance"] = "instance",
        device: str = "cuda",
    ):
        """Load a HuggingFace Transformers model.

        Args:
            model_path: Path to saved model directory
            model_type: Type of segmentation ("semantic" or "instance")
            device: Device to load model on

        Returns:
            VisionModel instance
        """
        from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

        # Load model and processor
        image_processor = AutoImageProcessor.from_pretrained(str(model_path))
        model = Mask2FormerForUniversalSegmentation.from_pretrained(str(model_path))
        model.to(device)
        model.eval()

        # Convert string to enum
        type_enum = (
            VisionModelType.SEMANTIC
            if model_type == "semantic"
            else VisionModelType.INSTANCE
        )

        instance = cls(type_enum, device=device)
        instance.model = model
        instance.image_processor = image_processor
        instance._backend = "hf"
        return instance
