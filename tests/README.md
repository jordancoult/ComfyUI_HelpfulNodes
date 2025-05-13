# Testing Utilities for CropAroundKPS Node

This directory contains helper scripts for the `CropAroundKPS` ComfyUI node. You can visualize crop results without requiring a full ComfyUI environment or a PyTorch installation.

## Files

- **run_crop_around_kps.py**
  - CLI script that loads a keypoints image, invokes the `CropAroundKPS` node logic, and draws the resulting crop rectangle on the image.
  - Usage:
    ```bash
    python run_crop_around_kps.py INPUT_IMAGE CROP_SIZE_MARGIN CROP_POS_MARGIN [-o OUTPUT]
    ```
  - Example:
    ```bash
    python run_crop_around_kps.py kpstest3.png 0.5 0.1
    # Produces kpstest3_cropped.png with a red crop box overlaid.
    ```
  - Supports a fallback dummy `torch` implementation if PyTorch is not installed.

## Dependencies
- Python 3.x
- [Pillow](https://pypi.org/project/Pillow/) (`pip install pillow`)
- [NumPy](https://pypi.org/project/numpy/) (`pip install numpy`)
- Optional: PyTorch. If not present, `run_crop_around_kps.py` uses a NumPy-based fallback.

## Running Tests

From the project root directory:

```bash
# Visual test on a sample image
python tests/run_crop_around_kps.py tests/kpstest3.png 0.5 0.1
```