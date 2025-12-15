from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset


def get_transforms(split="train", image_size=512):
    """Get augmentation transforms for semantic segmentation.

    Args:
        split: 'train' for augmented transforms, anything else for validation/test
        image_size: Target image size for resizing

    Returns:
        Albumentations Compose object
    """
    if split == "train":
        return A.Compose(
            [
                A.Resize(image_size, image_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5
                ),
                A.OneOf(
                    [
                        A.ElasticTransform(p=1),
                        A.GridDistortion(p=1),
                        A.OpticalDistortion(p=1),
                    ],
                    p=0.3,
                ),
                A.OneOf(
                    [
                        A.GaussNoise(p=1),
                        A.GaussianBlur(p=1),
                        A.MotionBlur(p=1),
                    ],
                    p=0.3,
                ),
                A.OneOf(
                    [
                        A.RandomBrightnessContrast(p=1),
                        A.HueSaturationValue(p=1),
                    ],
                    p=0.3,
                ),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(image_size, image_size),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )


def get_visualization_transforms(augment=False, image_size=512):
    """Get transforms for visualization (without normalization/ToTensor).

    Args:
        augment: Whether to apply data augmentations
        image_size: Target image size for resizing

    Returns:
        Albumentations Compose object
    """
    if augment:
        return A.Compose(
            [
                A.Resize(image_size, image_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5
                ),
                A.OneOf(
                    [
                        A.ElasticTransform(p=1),
                        A.GridDistortion(p=1),
                        A.OpticalDistortion(p=1),
                    ],
                    p=0.3,
                ),
                A.OneOf(
                    [
                        A.GaussNoise(p=1),
                        A.GaussianBlur(p=1),
                        A.MotionBlur(p=1),
                    ],
                    p=0.3,
                ),
                A.OneOf(
                    [
                        A.RandomBrightnessContrast(p=1),
                        A.HueSaturationValue(p=1),
                    ],
                    p=0.3,
                ),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(image_size, image_size),
            ]
        )


class YOLOSemanticDataset(Dataset):
    """Convert YOLO instance segmentation to semantic segmentation."""

    def __init__(self, dataset_path, split="train", transform=None):
        self.dataset_path = Path(dataset_path)
        self.images_dir = self.dataset_path / "images" / split
        self.labels_dir = self.dataset_path / "labels" / split
        self.transform = transform

        # Get all image files
        self.image_files = sorted(self.images_dir.glob("*.png"))

        if not self.image_files:
            raise ValueError(f"No images found in {self.images_dir}")

        print(f"Found {len(self.image_files)} images in {split} split")

    def __len__(self):
        return len(self.image_files)

    def _load_mask_from_yolo(self, label_path, img_shape):
        """Convert YOLO polygon format to binary mask."""
        mask = np.zeros(img_shape[:2], dtype=np.uint8)

        if not label_path.exists():
            return mask

        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                # Skip class_id (parts[0]), parse polygon coordinates
                coords = np.array(parts[1:], dtype=np.float32)
                coords = coords.reshape(-1, 2)

                # Convert normalized coordinates to absolute
                coords[:, 0] *= img_shape[1]  # x * width
                coords[:, 1] *= img_shape[0]  # y * height

                # Draw filled polygon
                cv2.fillPoly(mask, [coords.astype(np.int32)], 1)

        return mask

    def __getitem__(self, idx):
        # Load image
        image_path = self.image_files[idx]
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load mask
        label_path = self.labels_dir / (image_path.stem + ".txt")
        mask = self._load_mask_from_yolo(label_path, image.shape)

        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        # Convert to tensors if needed (when ToTensorV2 is not in the pipeline)
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).float()

        return image, mask.unsqueeze(0).float()
