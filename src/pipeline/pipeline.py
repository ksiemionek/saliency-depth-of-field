import cv2
import gen_saliency
import gen_depth
import gen_blur
from blend_maps import blend_maps
from utils import load_image


IMAGE_PATH = "./../../images/original.jpeg"

image = load_image(IMAGE_PATH)

saliency_model = gen_saliency.load_model(dense=True)
saliency = gen_saliency.generate_saliency(IMAGE_PATH, saliency_model)
cv2.imwrite("./results/saliency.png", saliency)

depth_processor, depth_model = gen_depth.load_model()
depth = gen_depth.generate_depth(image, depth_processor, depth_model)
cv2.imwrite("./results/depth.png", depth)

blur_mask = blend_maps(depth, saliency)
cv2.imwrite("./results/blur_mask.png", blur_mask)

blur_model = gen_blur.load_model()
bokeh = gen_blur.generate_blur(image, blur_mask, blur_model)
cv2.imwrite("./results/result.png", cv2.cvtColor(bokeh, cv2.COLOR_RGB2BGR))
