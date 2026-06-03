import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from backend.utils.device import get_torch_device


def load_model():
    device = get_torch_device()
    image_processor = AutoImageProcessor.from_pretrained(
        "depth-anything/Depth-Anything-V2-Large-hf"
    )
    model = AutoModelForDepthEstimation.from_pretrained(
        "depth-anything/Depth-Anything-V2-Large-hf"
    ).to(device)
    model.eval()

    return image_processor, model


def generate_depth(image, image_processor, model):
    device = get_torch_device()
    inputs = image_processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    post_processed_output = image_processor.post_process_depth_estimation(
        outputs,
        target_sizes=[(image.shape[0], image.shape[1])],
    )

    predicted_depth = post_processed_output[0]["predicted_depth"]

    depth = (predicted_depth - predicted_depth.min()) / (
        predicted_depth.max() - predicted_depth.min()
    )
    depth = depth.detach().cpu().numpy() * 255
    depth = depth.astype("uint8")

    return depth
