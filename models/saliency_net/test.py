import torch
from PIL import Image

from models import config
from models.saliency_net import model_config
from models.saliency_net.model import SaliencyNet
from models.saliency_net.transforms import image_transform
from models.utils.device import get_torch_device
from models.utils.image import save_saliency


def main() -> None:
    device = get_torch_device()

    model = SaliencyNet(model_config.BACKBONE, dropout=model_config.DROPOUT).to(device)
    model.load_state_dict(torch.load(config.CHECKPOINT_BEST, map_location=device))
    model.eval()

    img = Image.open(config.IMAGE).convert("RGB")
    x = image_transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(x).squeeze().cpu().numpy()

    save_saliency(pred, img.size, "saliency.png")


if __name__ == "__main__":
    main()
