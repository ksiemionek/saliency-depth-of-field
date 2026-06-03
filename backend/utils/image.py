import numpy as np
from PIL import Image


def load_image(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))
