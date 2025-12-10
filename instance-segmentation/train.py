from pathlib import Path
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForUniversalSegmentation, AutoImageProcessor
from dataset import YOLOSegmentationDataset
import numpy as np
from torchmetrics.detection.mean_ap import MeanAveragePrecision


# Configuration
CONFIG = {
    "dataset_path": "/home/jonas/Projects/vision-ai/datasets/yolo/housing-segmentation",
    "model_name": "facebook/mask2former-swin-tiny-coco-instance",
    "batch_size": 2,
    "num_epochs": 10,
    "learning_rate": 5e-5,
    "output_dir": "./outputs",
    "save_every": 10,
}


def collate_fn(batch):
    """Custom collate function for Mask2Former."""
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    mask_labels = [item["mask_labels"] for item in batch]
    class_labels = [item["class_labels"] for item in batch]
    
    return {
        "pixel_values": pixel_values,
        "mask_labels": mask_labels,
        "class_labels": class_labels,
    }


def compute_metrics(model, image_processor, dataloader, device):
    """Compute mAP metrics for instance segmentation."""
    metric = MeanAveragePrecision(iou_type="segm")
    model.eval()
    
    with torch.no_grad():
        for batch in dataloader:
            # Move pixel_values to device
            pixel_values = batch["pixel_values"].to(device)
            
            # Forward pass
            outputs = model(pixel_values=pixel_values)
            
            # Prepare targets - handle empty masks for background-only images
            target_sizes = []
            for mask in batch["mask_labels"]:
                if len(mask) > 0:
                    target_sizes.append((mask[0].shape[0], mask[0].shape[1]))
                else:
                    # For empty masks, use the original image size from pixel_values
                    target_sizes.append((pixel_values.shape[2], pixel_values.shape[3]))
            
            # Post-process predictions
            post_processed = image_processor.post_process_instance_segmentation(
                outputs,
                threshold=0.0,
                target_sizes=target_sizes,
                return_binary_maps=True,
            )
            
            # Collect predictions and targets
            predictions = []
            targets = []
            
            for i, (pred, target_masks, target_labels) in enumerate(
                zip(post_processed, batch["mask_labels"], batch["class_labels"])
            ):
                # Predictions
                if pred["segments_info"]:
                    predictions.append({
                        "masks": pred["segmentation"].to(dtype=torch.bool),
                        "labels": torch.tensor([x["label_id"] for x in pred["segments_info"]]),
                        "scores": torch.tensor([x["score"] for x in pred["segments_info"]]),
                    })
                else:
                    target_size = target_sizes[i]
                    predictions.append({
                        "masks": torch.zeros([0, *target_size], dtype=torch.bool),
                        "labels": torch.tensor([]),
                        "scores": torch.tensor([]),
                    })
                
                # Targets
                targets.append({
                    "masks": target_masks.to(dtype=torch.bool),
                    "labels": target_labels,
                })
            
            metric.update(predictions, targets)
    
    metrics = metric.compute()
    return {k: round(v.item(), 4) for k, v in metrics.items() if not k.endswith("per_class")}


def train():
    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load image processor
    image_processor = AutoImageProcessor.from_pretrained(
        CONFIG["model_name"],
        do_resize=True,
        size={"height": 512, "width": 512},
    )
    
    # Create label mappings
    label2id = {"bodenplatte": 0}
    id2label = {0: "bodenplatte"}
    
    # Load model
    model = AutoModelForUniversalSegmentation.from_pretrained(
        CONFIG["model_name"],
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
    )
    model.to(device)
    
    # Load datasets with image processor and augmentation for training
    train_dataset = YOLOSegmentationDataset(
        CONFIG["dataset_path"], 
        split="train",
        image_processor=image_processor,
        augment=True  # Enable augmentation for training
    )
    val_dataset = YOLOSegmentationDataset(
        CONFIG["dataset_path"], 
        split="valid",
        image_processor=image_processor
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"])
    
    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    
    # Training loop
    for epoch in range(CONFIG["num_epochs"]):
        model.train()
        train_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            # Move to device
            pixel_values = batch["pixel_values"].to(device)
            mask_labels = [m.to(device) for m in batch["mask_labels"]]
            class_labels = [c.to(device) for c in batch["class_labels"]]
            
            # Forward pass
            outputs = model(
                pixel_values=pixel_values,
                mask_labels=mask_labels,
                class_labels=class_labels,
            )
            loss = outputs.loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if (batch_idx + 1) % 5 == 0:
                print(f"Epoch [{epoch+1}/{CONFIG['num_epochs']}], "
                      f"Batch [{batch_idx+1}/{len(train_loader)}], "
                      f"Loss: {loss.item():.4f}")
        
        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{CONFIG['num_epochs']}] - Avg Train Loss: {avg_train_loss:.4f}")
        
        # Validation
        if (epoch + 1) % 5 == 0:
            print("Running validation...")
            metrics = compute_metrics(model, image_processor, val_loader, device)
            print(f"Validation Metrics: {metrics}")
        
        # Save checkpoint
        if (epoch + 1) % CONFIG["save_every"] == 0:
            checkpoint_path = output_dir / f"checkpoint_epoch_{epoch+1}.pt"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
    
    # Save final model
    final_path = output_dir / "final_model"
    model.save_pretrained(final_path)
    image_processor.save_pretrained(final_path)
    print(f"Final model saved to {final_path}")


if __name__ == "__main__":
    train()
