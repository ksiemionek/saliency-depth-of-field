from torchvision import transforms
import saliency_net.train_config as cfg


image_transform = transforms.Compose(
    [
        transforms.Resize(cfg.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.MEAN, std=cfg.STD),
    ]
)

heatmap_transform = transforms.Compose(
    [
        transforms.Resize(cfg.IMAGE_SIZE),
        transforms.ToTensor(),
    ]
)
