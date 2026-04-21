from transparent_background import Remover
from PIL import Image


def generate_saliency(image_path):
    remover = Remover()
    img = Image.open(image_path).convert('RGB')
    saliency_map = remover.process(img, type='map')

    return saliency_map


if __name__ == "__main__":
    saliency_map = generate_saliency(
        image_path="./../images/original.jpeg",
    )

    saliency_map.save("./result/saliency_object_oriented.png")