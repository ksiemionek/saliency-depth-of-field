import numpy as np
import torch
from PIL import Image

from backend import config
from backend.utils.device import get_torch_device
from models.saliency_net import model_config
from models.saliency_net.model import SaliencyNet
from models.saliency_net.transforms import image_transform


def load_model():
    device = get_torch_device()
    model = SaliencyNet(
        model_config.BACKBONE,
        dropout=model_config.DROPOUT,
        decoder_dim=model_config.DECODER_DIM,
    ).to(device)
    model.load_state_dict(torch.load(config.SALIENCY_CHECKPOINT, map_location=device))
    model.eval()

    return model


def generate_saliency(image_path: str, model) -> np.ndarray:
    device = get_torch_device()
    img = Image.open(image_path).convert("RGB")
    x = image_transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(x).squeeze().cpu().numpy()

    pred = (pred - pred.min()) / (pred.max() - pred.min())
    saliency = Image.fromarray((pred * 255).astype(np.uint8))
    saliency = saliency.resize(img.size)

    return np.asarray(saliency)
