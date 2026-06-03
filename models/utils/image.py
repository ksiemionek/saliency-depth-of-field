import numpy as np
from PIL import Image


def save_saliency(
    pred: np.ndarray,
    original_size: tuple[int, int],
    out_path: str,
) -> None:
    pred = (pred - pred.min()) / (pred.max() - pred.min())
    img = Image.fromarray((pred * 255).astype(np.uint8))
    img.resize(original_size).save(out_path)
