from transparent_background import Remover
from PIL import Image


remover = Remover()
img = Image.open("./original.jpeg").convert('RGB')
saliency_map = remover.process(img, type='map')
saliency_map.save("./saliency.png")
