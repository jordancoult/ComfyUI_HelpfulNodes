from . import nodes

# Register the node with the necessary mappings
NODE_CLASS_MAPPINGS = {
    "JCo_CropAroundKPS": nodes.CropAroundKPS
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JCo_CropAroundKPS": "Prep Crop Around Keypoints"
}