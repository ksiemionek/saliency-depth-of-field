import cv2
from backend import config
from backend.pipeline import gen_depth, gen_saliency, gen_blur, blend_maps
from backend.utils.image import load_image
import numpy as np
from PIL import Image


def main():
    image = load_image(config.IMAGE)

    saliency_model = gen_saliency.load_model(dense=True)
    saliency = gen_saliency.generate_saliency(config.IMAGE, saliency_model)
    # saliency = np.asarray(Image.open("saliency.png"))
    cv2.imwrite(config.PIPELINE_SALIENCY, saliency)

    depth_processor, depth_model = gen_depth.load_model()
    depth = gen_depth.generate_depth(image, depth_processor, depth_model)
    cv2.imwrite(config.PIPELINE_DEPTH, depth)

    blur_mask = blend_maps.blend_maps(depth, saliency)
    cv2.imwrite(config.PIPELINE_BLUR_MASK, blur_mask)

    blur_model = gen_blur.load_model()
    bokeh = gen_blur.generate_blur(image, blur_mask, blur_model)
    cv2.imwrite(config.PIPELINE_RESULT, cv2.cvtColor(bokeh, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    main()
