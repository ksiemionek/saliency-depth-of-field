import cv2
import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from device import device
from utils import load_image


def load_model():
    image_processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
    model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf").to(device)
    model.eval()

    return image_processor, model


def generate_depth(image, image_processor, model):
    inputs = image_processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    post_processed_output = image_processor.post_process_depth_estimation(
        outputs,
        target_sizes=[(image.shape[0], image.shape[1])],
    )

    predicted_depth = post_processed_output[0]["predicted_depth"]

    depth = (predicted_depth - predicted_depth.min()) / (predicted_depth.max() - predicted_depth.min())
    depth = depth.detach().cpu().numpy() * 255
    depth = depth.astype("uint8")

    return depth
