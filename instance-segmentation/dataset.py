import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset


class YOLOSegmentationDataset(Dataset):
    """YOLO polygon format dataset for Mask2Former instance segmentation."""

    def __init__(
        self, dataset_root, split="train", image_processor=None, augment=False
    ):
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.image_processor = image_processor
        self.augment = augment

        # Define augmentation pipeline
        self.augment = augment

        self.images_dir = self.dataset_root / "images" / split
        self.labels_dir = self.dataset_root / "labels" / split

        # Get all image files
        self.image_files = sorted([f for f in self.images_dir.glob("*.jpg")])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert("RGB")
        w, h = image.size

        # Load annotations
        label_path = self.labels_dir / f"{img_path.stem}.txt"

        # Create instance mask where each instance has unique ID
        instance_mask = np.zeros((h, w), dtype=np.uint16)
        instance_id_to_semantic_id = {}

        instance_id = 1  # Start from 1, 0 is background

        if label_path.exists() and label_path.stat().st_size > 0:
            # First pass: collect all polygons by class
            polygons_by_class = {}  # {class_id: [list of polygon arrays]}

            with open(label_path) as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split()
                if len(parts) < 3:  # Need at least class + 1 point
                    continue

                class_id = int(parts[0])
                # Remaining parts are polygon coordinates (normalized x, y pairs)
                coords = list(map(float, parts[1:]))

                # Convert to absolute coordinates
                polygon = []
                for i in range(0, len(coords), 2):
                    x = int(coords[i] * w)
                    y = int(coords[i + 1] * h)
                    polygon.append([x, y])

                # Store polygon for this class
                if class_id not in polygons_by_class:
                    polygons_by_class[class_id] = []
                polygons_by_class[class_id].append(np.array(polygon, dtype=np.int32))

            # Second pass: process each class's polygons together to handle holes
            for class_id, polygons in polygons_by_class.items():
                # Create a temporary mask for all polygons of this class
                temp_mask = np.zeros((h, w), dtype=np.uint8)

                # Draw all polygons with fillPoly
                cv2.fillPoly(temp_mask, polygons, color=255)

                # Find contours with hierarchy to identify holes
                contours, hierarchy = cv2.findContours(
                    temp_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
                )

                if hierarchy is None or len(contours) == 0:
                    continue

                # hierarchy format: [Next, Previous, First_Child, Parent]
                # Outer contours have Parent=-1, holes have Parent>=0
                hierarchy = hierarchy[0]  # Remove extra dimension

                # Process only top-level (outer) contours
                for i, contour in enumerate(contours):
                    if hierarchy[i][3] == -1:  # No parent = outer contour
                        # Create instance mask for this contour
                        contour_mask = np.zeros((h, w), dtype=np.uint8)
                        cv2.drawContours(contour_mask, [contour], -1, 255, cv2.FILLED)

                        # Find and subtract holes (children of this contour)
                        child_idx = hierarchy[i][2]  # First child index
                        while child_idx != -1:
                            # Subtract hole
                            hole_mask = np.zeros((h, w), dtype=np.uint8)
                            cv2.drawContours(
                                hole_mask, [contours[child_idx]], -1, 255, cv2.FILLED
                            )
                            contour_mask[hole_mask > 0] = 0

                            # Move to next sibling
                            child_idx = hierarchy[child_idx][0]

                        # Copy to instance mask with unique ID
                        instance_mask[contour_mask > 0] = instance_id

                        # Map instance ID to semantic class ID
                        instance_id_to_semantic_id[instance_id] = class_id
                        instance_id += 1

        # Apply augmentations if enabled (before processing)
        if self.augment:
            # Random horizontal flip
            if random.random() > 0.5:
                image = TF.hflip(image)
                instance_mask = np.fliplr(instance_mask)

            # Random vertical flip
            if random.random() > 0.5:
                image = TF.vflip(image)
                instance_mask = np.flipud(instance_mask)

            # Random rotation
            angle = random.uniform(-15, 15)
            image = TF.rotate(image, angle, interpolation=TF.InterpolationMode.BILINEAR)
            # Rotate mask using PIL
            mask_pil = Image.fromarray(instance_mask.astype(np.uint8))
            mask_pil = TF.rotate(
                mask_pil, angle, interpolation=TF.InterpolationMode.NEAREST
            )
            instance_mask = np.array(mask_pil)

            # Color jitter (only affects image, not mask)
            image = transforms.ColorJitter(
                brightness=0.5, contrast=0.3, saturation=0.5, hue=0.1
            )(image)

        # Process with image processor
        if self.image_processor:
            # For background-only images (no instances), create empty tensors
            if len(instance_id_to_semantic_id) == 0:
                # Process just the image without segmentation maps
                inputs = self.image_processor(images=image, return_tensors="pt")
                # Create empty masks and labels
                h_resized, w_resized = inputs.pixel_values.shape[-2:]
                return {
                    "pixel_values": inputs.pixel_values[0],
                    "mask_labels": torch.zeros(
                        (0, h_resized, w_resized), dtype=torch.float32
                    ),
                    "class_labels": torch.tensor([], dtype=torch.int64),
                }
            else:
                # Process image only (not segmentation maps) to avoid processor issues
                inputs = self.image_processor(images=image, return_tensors="pt")

                # Manually create masks and labels from instance_mask
                # Extract each instance as a separate binary mask
                unique_ids = sorted([id for id in instance_id_to_semantic_id])

                masks_list = []
                labels_list = []
                for inst_id in unique_ids:
                    # Create binary mask for this instance
                    binary_mask = (instance_mask == inst_id).astype(np.float32)
                    masks_list.append(binary_mask)
                    labels_list.append(instance_id_to_semantic_id[inst_id])

                if len(masks_list) > 0:
                    masks = np.stack(masks_list, axis=0)
                    labels = np.array(labels_list, dtype=np.int64)
                else:
                    masks = np.zeros((0, h, w), dtype=np.float32)
                    labels = np.array([], dtype=np.int64)

                # Resize masks to match processor output size
                h_resized, w_resized = inputs.pixel_values.shape[-2:]
                if masks.shape[0] > 0:
                    import torch.nn.functional as F

                    masks_tensor = torch.from_numpy(masks).unsqueeze(0)  # Add batch dim
                    masks_resized = F.interpolate(
                        masks_tensor, size=(h_resized, w_resized), mode="nearest"
                    )
                    masks = masks_resized.squeeze(0).numpy()  # Remove batch dim
                else:
                    masks = np.zeros((0, h_resized, w_resized), dtype=np.float32)

                return {
                    "pixel_values": inputs.pixel_values[0],
                    "mask_labels": torch.from_numpy(masks).float(),
                    "class_labels": torch.from_numpy(labels).long(),
                }
        else:
            return {
                "image": image,
                "instance_mask": instance_mask,
                "instance_id_to_semantic_id": instance_id_to_semantic_id,
            }
