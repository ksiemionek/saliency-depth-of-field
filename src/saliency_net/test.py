import torch
from PIL import Image
import config
import saliency_net.train_config as train_config
from saliency_net.model import SaliencyNet
from saliency_net.transforms import image_transform
from utils.device import get_torch_device
from utils.image import save_saliency


def main() -> None:
    device = get_torch_device()

    model = SaliencyNet(train_config.BACKBONE, dropout=train_config.DROPOUT).to(device)
    model.load_state_dict(torch.load(config.CHECKPOINT_BEST, map_location=device))
    model.eval()

    img = Image.open(config.IMAGE).convert("RGB")
    x = image_transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(x).squeeze().cpu().numpy()

    save_saliency(pred, img.size, "saliency_output.png")


if __name__ == "__main__":
    main()
