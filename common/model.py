from enum import Enum

import supervision as sv


class ModelType(Enum):
    SMP = "smp"


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


class SemanticSegmentationModel:
    def __init__(self, model_type: ModelType):
        self.model = None
        self.model_type = model_type
        self.transform = None

    def predict(self, image) -> sv.Detections:
        if self.model_type == ModelType.SMP:
            return self._predict_smp(image)

    def _predict_smp(self, image) -> sv.Detections:
        import numpy as np
        import torch

        device = next(self.model.parameters()).device

        # Apply transforms
        transformed = self.transform(image=image)
        image_tensor = transformed["image"].unsqueeze(0).to(device)

        # Make prediction
        with torch.inference_mode():
            output = self.model(image_tensor)
            mask = output.squeeze().cpu().numpy()

        # Threshold mask
        binary_mask = (mask > 0.5).astype(np.uint8)

        # Convert binary mask to detections
        return binary_mask_to_detections(binary_mask)

    @classmethod
    def from_smp(cls, model_path, device):
        import albumentations as A
        import segmentation_models_pytorch as smp

        model = smp.from_pretrained(str(model_path))
        model.to(device)
        model.eval()

        # Load transforms from saved model
        transform = A.Compose.from_pretrained(str(model_path))

        instance = cls(ModelType.SMP)
        instance.model = model
        instance.transform = transform
        return instance
