#!/usr/bin/env python3

"""
Command-line test script for CropAroundKPS node.

Usage:
  python run_crop_around_kps.py INPUT_IMAGE CROP_SIZE_MARGIN CROP_POS_MARGIN [-o OUTPUT]

Example:
  python run_crop_around_kps.py tests/kpstest3.png 0.5 0.1
  # Saves tests/kpstest3_cropped.png with a red rectangle marking the crop.
"""

import os
import sys
import argparse

# Ensure project root is in PYTHONPATH
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
sys.path.insert(0, project_root)

try:
    import torch
except ImportError:
    import numpy as _np
    import types
    import sys as _sys
    torch = types.ModuleType("torch")
    class DummyTensor:
        def __init__(self, array):
            self.array = array if isinstance(array, _np.ndarray) else _np.array(array)
        @property
        def shape(self):
            return self.array.shape
        def dim(self):
            return self.array.ndim
        def mean(self, dim):
            return DummyTensor(self.array.mean(axis=dim))
        def __getitem__(self, key):
            return DummyTensor(self.array[key])
        def unsqueeze(self, axis):
            return DummyTensor(_np.expand_dims(self.array, axis))
        def float(self):
            return self
        def size(self, dim):
            return self.array.shape[dim]
        def __gt__(self, other):
            rhs = other.array if isinstance(other, DummyTensor) else other
            return DummyTensor(self.array > rhs)
    def from_numpy(np_array):
        return DummyTensor(np_array)
    def nonzero(cond, as_tuple=False):
        arr = cond.array if isinstance(cond, DummyTensor) else cond
        coords = _np.argwhere(arr)
        return DummyTensor(coords)
    def min_fn(tensor, dim):
        vals = tensor.array.min(axis=dim)
        return types.SimpleNamespace(values=vals)
    def max_fn(tensor, dim):
        vals = tensor.array.max(axis=dim)
        return types.SimpleNamespace(values=vals)
    torch.from_numpy = from_numpy
    torch.nonzero = nonzero
    torch.min = min_fn
    torch.max = max_fn
    _sys.modules["torch"] = torch
import numpy as np
from PIL import Image, ImageDraw

from nodes import CropAroundKPS

def main():
    parser = argparse.ArgumentParser(
        description="Draw the crop rectangle computed by CropAroundKPS on the input image."
    )
    parser.add_argument("input_image", help="Path to input keypoints image (PNG, JPEG, etc.)")
    parser.add_argument("crop_size_margin", type=float,
                        help="Crop size margin (percentage of bbox longest side, e.g. 0.5)")
    parser.add_argument("crop_pos_margin", type=float,
                        help="Crop position margin (percentage of crop height, e.g. 0.1)")
    parser.add_argument("-o", "--output", default=None,
                        help="Path to save output image; defaults to INPUT_CROPPED.png")
    args = parser.parse_args()

    # Load image and convert to RGB
    img = Image.open(args.input_image).convert("RGB")
    img_np = np.array(img)
    h, w, c = img_np.shape

    # Create tensor of shape (1, H, W, C)
    img_tensor = torch.from_numpy(img_np).unsqueeze(0).float()

    # Compute crop rectangle
    node = CropAroundKPS()
    width, height, x, y = node.crop_around_keypoints(
        img_tensor, args.crop_size_margin, args.crop_pos_margin
    )

    # Draw rectangle on the image
    draw = ImageDraw.Draw(img)
    draw.rectangle([x, y, x + width, y + height], outline="red", width=3)

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        base, ext = os.path.splitext(args.input_image)
        output_path = f"{base}_cropped{ext}"

    # Save and report
    img.save(output_path)
    print(f"Output saved to: {output_path}")

if __name__ == "__main__":
    main()