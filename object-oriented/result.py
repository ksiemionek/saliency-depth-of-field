import cv2
import numpy as np


def blur(img, depth, saliency, output, max_blur):
    img = cv2.imread(img)
    depth = cv2.imread(depth, cv2.IMREAD_GRAYSCALE) / 255.0
    saliency = cv2.imread(saliency, cv2.IMREAD_GRAYSCALE) / 255.0

    blur_mask = ((1.0 - depth) ** 0.5) * (1.0 - saliency)
    blur_mask = np.clip(blur_mask, 0.0, 1.0)
    blur_mask = np.expand_dims(blur_mask, axis=-1)
    cv2.imwrite("blur_mask.png", (blur_mask * 255).astype(np.uint8))

    blur_mask_inv = 1.0 - blur_mask
    cv2.imwrite('blur_mask_inv.png', (blur_mask_inv * 255).astype(np.uint8))

    k = max(3, max_blur)
    blurred_img = cv2.GaussianBlur(img, (k, k), 0).astype(np.float32)
    result = img * (1.0 - blur_mask) + blurred_img * blur_mask

    cv2.imwrite(output, result.astype(np.uint8))


blur(
    "./original.jpeg",
    "./depth.png",
    "./saliency.png",
    "result.png",
    75,
)
