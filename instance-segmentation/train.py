from dataclasses import dataclass
from pathlib import Path

import torch
from dataset import YOLOSegmentationDataset
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from transformers import (
    AutoImageProcessor,
    AutoModelForUniversalSegmentation,
    Trainer,
    TrainingArguments,
)

# Resolve dataset path relative to this script
SCRIPT_DIR = Path(__file__).parent
DATASET_PATH = (SCRIPT_DIR / "../datasets/yolo/donut").resolve()

# Configuration
CONFIG = {
    "model_name": "facebook/mask2former-swin-tiny-coco-instance",
    "output_dir": "./outputs",
}


@dataclass
class ModelOutput:
    """Wrapper for model outputs to match expected format."""

    class_queries_logits: torch.Tensor
    masks_queries_logits: torch.Tensor


def collate_fn(batch):
    """Custom collate function for Mask2Former."""
    return {
        "pixel_values": torch.stack([item["pixel_values"] for item in batch]),
        "mask_labels": [item["mask_labels"] for item in batch],
        "class_labels": [item["class_labels"] for item in batch],
    }


class Evaluator:
    """Compute metrics for instance segmentation."""

    def __init__(self, image_processor, id2label):
        self.image_processor = image_processor
        self.id2label = id2label
        self.metric = MeanAveragePrecision(iou_type="segm", class_metrics=True)

    def __call__(self, eval_pred):
        """Compute metrics from predictions and labels."""
        predictions, labels = eval_pred

        # predictions is a tuple of (class_queries_logits, masks_queries_logits)
        # from model forward pass
        class_queries_logits = torch.from_numpy(predictions[0])
        masks_queries_logits = torch.from_numpy(predictions[1])

        mask_labels, class_labels = labels

        # Prepare target sizes
        target_sizes = [masks[0].shape[-2:] for masks in mask_labels]

        # Wrap outputs in ModelOutput dataclass
        model_outputs = ModelOutput(
            class_queries_logits=class_queries_logits,
            masks_queries_logits=masks_queries_logits,
        )

        # Post-process predictions
        post_processed = self.image_processor.post_process_instance_segmentation(
            outputs=model_outputs,
            threshold=0.0,
            target_sizes=target_sizes,
            return_binary_maps=True,
        )

        # Format predictions and targets for metric computation
        formatted_predictions = []
        formatted_targets = []

        for pred, target_masks, target_labels, target_size in zip(
            post_processed, mask_labels, class_labels, target_sizes, strict=True
        ):
            if pred["segments_info"]:
                formatted_predictions.append(
                    {
                        "masks": pred["segmentation"].to(dtype=torch.bool),
                        "labels": torch.tensor(
                            [x["label_id"] for x in pred["segments_info"]]
                        ),
                        "scores": torch.tensor(
                            [x["score"] for x in pred["segments_info"]]
                        ),
                    }
                )
            else:
                formatted_predictions.append(
                    {
                        "masks": torch.zeros([0, *target_size], dtype=torch.bool),
                        "labels": torch.tensor([]),
                        "scores": torch.tensor([]),
                    }
                )

            formatted_targets.append(
                {
                    "masks": torch.from_numpy(target_masks).to(dtype=torch.bool),
                    "labels": torch.from_numpy(target_labels),
                }
            )

        # Update and compute metrics
        self.metric.update(formatted_predictions, formatted_targets)
        metrics = self.metric.compute()

        # Flatten per-class metrics - handle both single and multiple classes
        classes = metrics.pop("classes")
        map_per_class = metrics.pop("map_per_class")
        mar_100_per_class = metrics.pop("mar_100_per_class")

        # Ensure they are iterable (convert 0-d tensors to 1-d)
        if classes.dim() == 0:
            classes = classes.unsqueeze(0)
            map_per_class = map_per_class.unsqueeze(0)
            mar_100_per_class = mar_100_per_class.unsqueeze(0)

        for class_id, class_map, class_mar in zip(
            classes, map_per_class, mar_100_per_class, strict=True
        ):
            class_name = self.id2label.get(class_id.item(), class_id.item())
            metrics[f"map_{class_name}"] = class_map
            metrics[f"mar_100_{class_name}"] = class_mar

        # Reset for next evaluation
        self.metric.reset()

        return {k: round(v.item(), 4) for k, v in metrics.items()}


def main():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Label mappings
    label2id = {"donut": 0}
    id2label = {0: "donut"}

    # Load model and image processor
    model = AutoModelForUniversalSegmentation.from_pretrained(
        CONFIG["model_name"],
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
    )

    image_processor = AutoImageProcessor.from_pretrained(
        CONFIG["model_name"],
        do_resize=True,
        size={"height": 512, "width": 512},
    )

    # Load datasets
    train_dataset = YOLOSegmentationDataset(
        str(DATASET_PATH),
        split="train",
        image_processor=image_processor,
        augment=True,
    )

    val_dataset = YOLOSegmentationDataset(
        str(DATASET_PATH),
        split="valid",
        image_processor=image_processor,
        augment=False,
    )

    print(
        f"Training samples: {len(train_dataset)}, "
        f"Validation samples: {len(val_dataset)}"
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=CONFIG["output_dir"],
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        learning_rate=5e-5,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=5,
        remove_unused_columns=False,
        push_to_hub=False,
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=image_processor,
        data_collator=collate_fn,
        compute_metrics=Evaluator(image_processor=image_processor, id2label=id2label),
    )

    # Train
    trainer.train()

    # Save final model
    trainer.save_model()
    print("Training completed!")


if __name__ == "__main__":
    main()
