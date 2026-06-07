from torchvision import transforms
from torchvision.transforms import InterpolationMode

from models.saliency_net import model_config

image_transform = transforms.Compose(
    [
        transforms.Resize(model_config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=model_config.MEAN, std=model_config.STD),
    ]
)

heatmap_transform = transforms.Compose(
    [
        transforms.Resize(model_config.IMAGE_SIZE),
        transforms.ToTensor(),
    ]
)

fixation_transform = transforms.Compose(
    [
        transforms.Resize(
            model_config.IMAGE_SIZE, interpolation=InterpolationMode.NEAREST
        ),
        transforms.ToTensor(),
    ]
)
