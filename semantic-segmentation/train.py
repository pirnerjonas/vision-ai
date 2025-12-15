from pathlib import Path

import segmentation_models_pytorch as smp
import torch
from dataset import YOLOSemanticDataset, get_transforms
from torch.utils.data import DataLoader

# ==================== CONFIGURATION ====================
SCRIPT_DIR = Path(__file__).parent
DATASET_PATH = (SCRIPT_DIR / "../datasets/yolo/zinnperle-segmentation").resolve()
OUTPUT_DIR = SCRIPT_DIR / "outputs"

CONFIG = {
    "encoder": "resnet34",
    "encoder_weights": "imagenet",
    "classes": 1,
    "activation": "sigmoid",
    "learning_rate": 0.0001,
    "epochs": 200,
    "batch_size": 8,
    "image_size": 512,
}
# =======================================================


def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.inference_mode()
def validate(model, loader, criterion, device):
    """Validate model with detailed metrics."""
    model.eval()
    total_loss = 0
    total_iou = 0
    total_dice = 0
    total_precision = 0
    total_recall = 0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)
        loss = criterion(outputs, masks)
        total_loss += loss.item()

        # Calculate metrics
        preds = (outputs > 0.5).float()

        # IoU
        intersection = (preds * masks).sum()
        union = preds.sum() + masks.sum() - intersection
        iou = (intersection + 1e-6) / (union + 1e-6)
        total_iou += iou.item()

        # Dice coefficient
        dice = (2 * intersection + 1e-6) / (preds.sum() + masks.sum() + 1e-6)
        total_dice += dice.item()

        # Precision and Recall
        true_positive = (preds * masks).sum()
        predicted_positive = preds.sum()
        actual_positive = masks.sum()

        precision = (true_positive + 1e-6) / (predicted_positive + 1e-6)
        recall = (true_positive + 1e-6) / (actual_positive + 1e-6)

        total_precision += precision.item()
        total_recall += recall.item()

    n = len(loader)
    return {
        "loss": total_loss / n,
        "iou": total_iou / n,
        "dice": total_dice / n,
        "precision": total_precision / n,
        "recall": total_recall / n,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Dataset path: {DATASET_PATH}")

    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Load datasets (no train/val split, using all data for training)
    train_dataset = YOLOSemanticDataset(
        DATASET_PATH,
        split="train",
        transform=get_transforms("train", CONFIG["image_size"]),
    )

    val_dataset = YOLOSemanticDataset(
        DATASET_PATH,
        split="valid",
        transform=get_transforms("valid", CONFIG["image_size"]),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=4,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=4,
    )

    # Create model
    model = smp.Unet(
        encoder_name=CONFIG["encoder"],
        encoder_weights=CONFIG["encoder_weights"],
        classes=CONFIG["classes"],
        activation=CONFIG["activation"],
    )
    model = model.to(device)

    # Setup training
    criterion = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])

    print(f"Training for {CONFIG['epochs']} epochs...")

    # Training loop
    best_iou = 0
    for epoch in range(CONFIG["epochs"]):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = validate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch + 1}/{CONFIG['epochs']}: "
            f"train_loss={train_loss:.4f}, "
            f"val_loss={val_metrics['loss']:.4f}, "
            f"val_iou={val_metrics['iou']:.4f}, "
            f"val_dice={val_metrics['dice']:.4f}, "
            f"val_precision={val_metrics['precision']:.4f}, "
            f"val_recall={val_metrics['recall']:.4f}"
        )

        # Save best model
        if val_metrics["iou"] > best_iou:
            best_iou = val_metrics["iou"]
            torch.save(model.state_dict(), OUTPUT_DIR / "best_model.pth")
            print(f"✓ Saved best model with IoU: {best_iou:.4f}")

    # Save final model
    torch.save(model.state_dict(), OUTPUT_DIR / "final_model.pth")
    print(f"\nTraining completed! Best IoU: {best_iou:.4f}")
    print(f"Models saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
