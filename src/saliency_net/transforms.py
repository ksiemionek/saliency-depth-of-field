from torchvision import transforms
import saliency_net.train_config as train_config


image_transform = transforms.Compose(
    [
        transforms.Resize(train_config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=train_config.MEAN, std=train_config.STD),
    ]
)

heatmap_transform = transforms.Compose(
    [
        transforms.Resize(train_config.IMAGE_SIZE),
        transforms.ToTensor(),
    ]
)
