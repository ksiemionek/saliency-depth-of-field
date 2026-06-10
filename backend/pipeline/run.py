import cv2
import numpy as np

from backend import config
from backend.pipeline import blend_maps, gen_blur, gen_depth, gen_saliency
from backend.pipeline.models import Models


def run_pipeline(image: np.ndarray, models: Models) -> np.ndarray:
    saliency = gen_saliency.generate_saliency(image, models.saliency_model)
    cv2.imwrite(config.PIPELINE_SALIENCY, saliency)

    depth = gen_depth.generate_depth(image, models.depth_processor, models.depth_model)
    cv2.imwrite(config.PIPELINE_DEPTH, depth)

    blur_mask = blend_maps.blend_maps(depth, saliency)
    cv2.imwrite(config.PIPELINE_BLUR_MASK, blur_mask)

    bokeh = gen_blur.generate_blur(image, blur_mask, models.blur_model)
    cv2.imwrite(config.PIPELINE_RESULT, cv2.cvtColor(bokeh, cv2.COLOR_RGB2BGR))

    return bokeh
