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


def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model and processor
    model = AutoModelForUniversalSegmentation.from_pretrained(CONFIG["model_path"])
    image_processor = AutoImageProcessor.from_pretrained(CONFIG["model_path"])
    model.to(device)
    model.eval()

    # Get test images
    test_images_dir = Path(CONFIG["dataset_path"]) / "images" / "test"
    image_files = sorted(list(test_images_dir.glob("*.jpg")))

    # Create output directory
    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Annotator for visualization
    mask_annotator = sv.MaskAnnotator()

    print(f"Processing {len(image_files)} test images...")

    with torch.no_grad():
        for img_path in image_files:
            # Load image
            image = Image.open(img_path).convert("RGB")

            # Preprocess
            inputs = image_processor(images=image, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Predict
            outputs = model(**inputs)

            # Post-process
            target_size = (image.height, image.width)
            results = image_processor.post_process_instance_segmentation(
                outputs,
                threshold=0.5,
                target_sizes=[target_size],
                return_binary_maps=True,
            )[0]

            # Convert to supervision format
            if results["segments_info"]:
                masks = results["segmentation"].cpu().numpy().astype(bool)
                detections = sv.Detections(
                    xyxy=sv.mask_to_xyxy(masks),
                    mask=masks,
                    class_id=np.array(
                        [seg["label_id"] for seg in results["segments_info"]]
                    ),
                    confidence=np.array(
                        [seg["score"] for seg in results["segments_info"]]
                    ),
                )

                # Annotate image
                annotated_image = mask_annotator.annotate(
                    scene=np.array(image), detections=detections
                )
            else:
                # No detections
                annotated_image = np.array(image)

            # Save
            output_path = output_dir / f"{img_path.stem}_pred.jpg"
            Image.fromarray(annotated_image).save(output_path)
            print(f"Saved: {output_path}")

    print(f"All predictions saved to {output_dir}")


if __name__ == "__main__":
    test()
