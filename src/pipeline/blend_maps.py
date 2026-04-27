import numpy as np

def blend_maps(depth, saliency):
    depth = depth.astype(np.float32) / 255.0
    saliency = saliency.astype(np.float32) / 255.0

    blur_mask = np.clip(depth + saliency, 0.0, 1.0)
    blur_mask = (blur_mask * 255).astype(np.uint8)
    
    return blur_mask
