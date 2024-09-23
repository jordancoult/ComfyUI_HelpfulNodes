import torch

class CropAroundKPS:

    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "crop_size_margin": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.05,
                    "round": 0.01,
                    "display": "number"
                }),
                "crop_pos_margin": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.05,
                    "round": 0.01,
                    "display": "number"
                }),
            },
        }

    RETURN_TYPES = ("INT", "INT", "INT", "INT",)
    RETURN_NAMES = ("width", "height", "x", "y",)
    FUNCTION = "crop_around_keypoints"

    CATEGORY = "Keypoints Helpers"

    # In ComfyUI, the data exchanged as IMAGE type is always a 4-dimensional torch.tensor with dimensions (b, h, w, c)
    def crop_around_keypoints(self, image, crop_size_margin, crop_pos_margin):
        print(f"Received image shape: {image.shape}")
        print(f"Received crop_size_margin: {crop_size_margin}")

        UNCROPPED_IMAGE = (image.shape[1], image.shape[0], 0, 0)  # can return found_bbox bool flag with this appended

        if crop_pos_margin > crop_size_margin:
            print(f"Cannot have crop_pos_margin > crop_size_margin. Setting crop_pos_margin to crop_size_margin. crop_pos_margin: {crop_pos_margin}, crop_size_margin: {crop_size_margin}")
            crop_pos_margin = crop_size_margin
        
        # Ensure the image tensor is 4-dimensional and has shape (b, h, w, c)
        if image.dim() != 4:
            raise ValueError("Input image must be a 4-dimensional tensor with shape (b, h, w, c)")

        # Handle batch size > 1, we just process the first image in the batch
        image = image[0]
        print(f"Processing first image in batch, new shape: {image.shape}")

        # Convert to grayscale by averaging the channels, assuming the last dimension is channel
        grayscale = image[..., :3].mean(dim=-1)
        print(f"Grayscale image shape: {grayscale.shape}")

        # Find non-black pixels (values greater than 0)
        non_black_pixels = torch.nonzero(grayscale > 0, as_tuple=False)
        print(f"Number of non-black pixels found: {non_black_pixels.size(0)}")

        if non_black_pixels.size(0) == 0:
            # If no non-black pixels are found, return uncropped image
            print("No non-black pixels found, returning zeroed bounding box.")
            return UNCROPPED_IMAGE

        # Get the coordinates of the bounding box
        bbox_y_min, bbox_x_min = torch.min(non_black_pixels, dim=0).values
        bbox_y_max, bbox_x_max = torch.max(non_black_pixels, dim=0).values
        print(f"Bounding box - bbox_x_min: {bbox_x_min}, bbox_y_min: {bbox_y_min}, bbox_x_max: {bbox_x_max}, bbox_y_max: {bbox_y_max}")

        # Calculate bbox width and height
        bbox_width = bbox_x_max - bbox_x_min + 1
        bbox_height = bbox_y_max - bbox_y_min + 1
        largest_side = max(bbox_width, bbox_height)
        print(f"BBox width: {bbox_width}, height: {bbox_height}, largest_side: {largest_side}")

        # Calculate new size and crop coordinates from widths
        new_total_width = int(largest_side + (2 * largest_side * crop_size_margin))
        new_total_height = int(new_total_width * (image.shape[0] / image.shape[1]))
        print(f"New total width: {new_total_width}, New total height: {new_total_height}")

        # Calculate top-left coordinates of new width/height such that the new w/h is centered in the original image
        new_x = int((image.shape[1] - new_total_width) / 2)
        new_y = int((image.shape[0] - new_total_height) / 2)
        print(f"Top-left corner coordinates - x: {new_x}, y: {new_y}")

        # If x or y are negative then they should both be 0 and width/height should be equal to totals
        if new_x < 0 or new_y < 0:
            print(f"Top-left corner coordinates - new_x: {new_x}, new_y: {new_y}. Resetting width and height to image size so no crop is performed.")
            return UNCROPPED_IMAGE

        # Move crop so it encompasses the bounding box, constraining to original image bounds. Add margin around bbox if possible
        CROP_MARGIN = int(new_total_height * crop_pos_margin)  # Pixels. A percent of crop height.
        if (bbox_x_min < new_x):
            new_x = max(0, bbox_x_min - CROP_MARGIN)
        if (bbox_y_min < new_y):
            new_y = max(0, bbox_y_min - CROP_MARGIN)
        if (bbox_x_max > new_x + new_total_width):
            new_x = min(image.shape[1] - new_total_width, bbox_x_max - new_total_width + CROP_MARGIN)
        if (bbox_y_max > new_y + new_total_height):
            new_y = min(image.shape[0] - new_total_height, bbox_y_max - new_total_height + CROP_MARGIN)

        # Double check that the bbox is within crop. If it's not, print and return uncropped image
        if (bbox_x_min < new_x or bbox_y_min < new_y or bbox_x_max > new_x + new_total_width or bbox_y_max > new_y + new_total_height):
            print(f"Bounding box not within crop - bbox_x_min: {bbox_x_min}, bbox_y_min: {bbox_y_min}, bbox_x_max: {bbox_x_max}, bbox_y_max: {bbox_y_max}")
            print(f"Top-left corner coordinates - new_x: {new_x}, new_y: {new_y}. Resetting width and height to image size so no crop is performed.")
            return UNCROPPED_IMAGE
        
        # Return new crop width, height, x, y
        print(f"Returning values - width: {new_total_width}, height: {new_total_height}, new_x: {new_x}, new_y: {new_y}")
        return (new_total_width, new_total_height, new_x, new_y)

# Register the node with the necessary mappings
NODE_CLASS_MAPPINGS = {
    "CropAroundKPS": CropAroundKPS
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CropAroundKPS": "Prepare Crop Around Keypoints"
}
