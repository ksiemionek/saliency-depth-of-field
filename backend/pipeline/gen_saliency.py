import numpy as np
import torch
from torchvision import transforms
from backend.pipeline.TranSalNet.utils.data_process import postprocess_img, preprocess_img
from backend.utils.device import get_torch_device
from backend import config


def load_model(dense=True):
    if dense:
        from backend.pipeline.TranSalNet.TranSalNet_Dense import TranSalNet
    else:
        from backend.pipeline.TranSalNet.TranSalNet_Res import TranSalNet

    device = get_torch_device()
    weight_path = (
        f"{config.TRANSALNET}/{f"TranSalNet_{'Dense' if dense else 'Res'}.pth"}"
    )

    model = TranSalNet()
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model = model.to(device)
    model.eval()

    return model


def generate_saliency(image_path, model):
    device = get_torch_device()
    img = preprocess_img(image_path)
    img = np.array(img) / 255.0
    img = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0)
    img = torch.from_numpy(img).float().to(device)

    with torch.no_grad():
        pred = model(img)

    to_pil = transforms.ToPILImage()
    pred_pil = to_pil(pred.squeeze().cpu())

    saliency_map = postprocess_img(pred_pil, image_path)

    return saliency_map
