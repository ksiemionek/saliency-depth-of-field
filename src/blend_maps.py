import numpy as np
import cv2

from gen_saliency import (
    load_model as load_saliency_model,
    generate_saliency,
)
from gen_depth import (
    load_model as load_depth_model,
    generate_depth,
)


def blend_maps(depth, saliency):
    depth = depth.astype(np.float32) / 255.0
    saliency = saliency.astype(np.float32) / 255.0

    blur_mask = 1.0 - (1.0 - depth) * (1.0 - saliency)
    blur_mask = (blur_mask * 255).astype(np.uint8)
    
    return blur_mask


def run(image_path, output_path, dense=True, alpha=0.7):
    saliency_model = load_saliency_model(dense=dense)
    depth_processor, depth_model = load_depth_model()

    saliency = generate_saliency(image_path, saliency_model)
    depth = generate_depth(image_path, depth_processor, depth_model)
    blur_mask = blend_maps(depth, saliency)

    cv2.imwrite(output_path, blur_mask, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])

    # return saliency, depth, blur_mask


if __name__ == "__main__":
    run(
        image_path="./../images/original.jpeg",
        output_path="./results/blur_mask.png",
        dense=True
    )