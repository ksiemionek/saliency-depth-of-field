from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def imagenet_tensor_transform(size: tuple[int, int] = (256, 256)):
    return transforms.Compose(
        [
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def heatmap_tensor_transform(size: tuple[int, int] = (256, 256)):
    return transforms.Compose(
        [
            transforms.Resize(size),
            transforms.ToTensor(),
        ]
    )
