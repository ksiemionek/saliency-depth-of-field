import cv2
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from device import device


def load_model():
    image_processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
    model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf").to(device)
    model.eval()

    return image_processor, model


def generate_depth(image_path, image_processor, model):
    image = Image.open(image_path)

    inputs = image_processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    post_processed_output = image_processor.post_process_depth_estimation(
        outputs,
        target_sizes=[(image.height, image.width)],
    )

    predicted_depth = post_processed_output[0]["predicted_depth"]

    depth = (predicted_depth - predicted_depth.min()) / (predicted_depth.max() - predicted_depth.min())
    depth = depth.detach().cpu().numpy() * 255
    depth = depth.astype("uint8")

    return depth


if __name__ == "__main__":
    image_processor, model = load_model()

    depth = generate_depth(
        image_path="./../images/original.jpeg",
        image_processor=image_processor,
        model=model,
    )

    cv2.imwrite("./results/depth_anything.png", depth, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
