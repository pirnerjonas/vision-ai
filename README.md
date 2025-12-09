# Vision AI - Instance Segmentation

## Mask2Former Training on YOLO Segmentation Dataset

### Key Findings

**YOLO Polygon Format with Holes:**
- YOLO stores outer contours and holes as separate polygons (separate lines in label files)
- Each polygon has the same class_id
- Solution: Group polygons by class, use `cv2.findContours(RETR_TREE)` to get hierarchy, identify outer contours (parent=-1), and subtract child contours (holes)

**Mask2Former Data Processing:**
- The built-in image processor's `convert_segmentation_map_to_binary_masks` has issues with instance segmentation
- Using `do_reduce_labels=True` causes class IDs to become -1 (CUDA assertion error)
- Using `do_reduce_labels=False` causes KeyError when looking up background pixels
- Solution: Process images with processor (for normalization), but manually create binary masks and resize them to match

**Implementation Details:**
- Bypassed processor's segmentation conversion
- Created binary masks directly from instance IDs
- Used `torch.nn.functional.interpolate` to resize masks
- Result: Training works correctly with proper hole handling

### Notes
- Mask2Former can handle empty annotations (background-only images) when properly handled
- Loss decreases nicely from ~40 to ~11 over initial epochs