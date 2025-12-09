from pathlib import Path
import torch
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor
from dataset import YOLOSegmentationDataset
import numpy as np
import supervision as sv
from PIL import Image


# Configuration
CONFIG = {
    "dataset_path": "/home/jonas/Projects/vision-ai/datasets/yolo/housing-segmentation",
    "split": "train",  # Can be "train", "valid", or "test"
    "output_dir": "./visualizations",
    "num_samples": 10,  # Number of samples to visualize
}


def collate_fn(batch):
    """Custom collate function."""
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    mask_labels = [item["mask_labels"] for item in batch]
    class_labels = [item["class_labels"] for item in batch]
    
    return {
        "pixel_values": pixel_values,
        "mask_labels": mask_labels,
        "class_labels": class_labels,
    }


def visualize():
    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load image processor
    image_processor = AutoImageProcessor.from_pretrained(
        "facebook/mask2former-swin-tiny-coco-instance",
        do_resize=True,
        size={"height": 512, "width": 512},
    )
    
    # Load dataset
    dataset = YOLOSegmentationDataset(
        CONFIG["dataset_path"],
        split=CONFIG["split"],
        image_processor=image_processor
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
    )
    
    # Annotator for visualization
    mask_annotator = sv.MaskAnnotator()
    
    print(f"Visualizing {min(CONFIG['num_samples'], len(dataset))} samples from {CONFIG['split']} set...")
    
    for idx, batch in enumerate(dataloader):
        if idx >= CONFIG["num_samples"]:
            break
        
        # Get data
        pixel_values = batch["pixel_values"][0]
        mask_labels = batch["mask_labels"][0]
        class_labels = batch["class_labels"][0]
        
        # Denormalize image (reverse ImageNet normalization)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image = pixel_values * std + mean
        image = (image * 255).clamp(0, 255).permute(1, 2, 0).numpy().astype(np.uint8)
        image = np.ascontiguousarray(image)  # Ensure contiguous memory layout for OpenCV
        
        # Check if there are any instances
        if len(mask_labels) > 0:
            # Convert masks to boolean (values should already be 0 or 1)
            masks = mask_labels.numpy().astype(bool)
            
            # Debug: check if masks are actually different and check for holes
            print(f"  Sample {idx}: {len(mask_labels)} instances")
            
            # Create supervision Detections object for visualization
            detections = sv.Detections(
                xyxy=sv.mask_to_xyxy(masks=masks),
                mask=masks,
                class_id=class_labels.numpy(),  # Use class IDs as-is (do_reduce_labels=False)
            )
            
            # Annotate image
            annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
        else:
            # No instances (background only)
            print(f"  Sample {idx}: No instances (background only)")
            annotated_image = image
        
        # Save
        output_path = output_dir / f"{CONFIG['split']}_sample_{idx:03d}.jpg"
        Image.fromarray(annotated_image).save(output_path)
        print(f"Saved: {output_path} ({len(class_labels)} instances)")
    
    print(f"All visualizations saved to {output_dir}")


if __name__ == "__main__":
    visualize()
