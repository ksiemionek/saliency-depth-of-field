import numpy as np
import torch
from torchvision import transforms
from TranSalNet.utils.data_process import preprocess_img, postprocess_img
from device import device


def load_model(dense=True):
    if dense:
        from TranSalNet.TranSalNet_Dense import TranSalNet
    else:
        from TranSalNet.TranSalNet_Res import TranSalNet

    model = TranSalNet()
    model.load_state_dict(torch.load(
        f"./TranSalNet/pretrained_models/TranSalNet_{'Dense' if dense else 'Res'}.pth",
        map_location=device
    ))
    model = model.to(device)
    model.eval()

    return model


def generate_saliency(image_path, model):
    img = preprocess_img(image_path)
    img = np.array(img) / 255.
    img = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0)
    img = torch.from_numpy(img).float().to(device)

    with torch.no_grad():
        pred = model(img)

    to_pil = transforms.ToPILImage()
    pred_pil = to_pil(pred.squeeze().cpu())

    saliency_map = postprocess_img(pred_pil, image_path)

    return saliency_map
