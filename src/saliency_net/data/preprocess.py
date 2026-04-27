import scipy.io
import numpy as np
from PIL import Image
from pathlib import Path
from scipy.ndimage import gaussian_filter

MAT_DIR = Path('./DUT-OMRON/DUT-OMRON-eye-fixations/mat')
IMG_DIR = Path('./DUT-OMRON/DUT-OMRON-image')
OUT_DIR = Path('./DUT-OMRON/DUT-OMRON-heatmaps')

for mat_path in MAT_DIR.glob('*.mat'):
    img_path = IMG_DIR / mat_path.with_suffix('.jpg').name
    
    img = np.asarray(Image.open(img_path))
    
    data = scipy.io.loadmat(mat_path)
    points = data['s']
    
    heatmap = np.zeros((img.shape[0],  img.shape[1]), dtype=np.float32)
    for x, y, _ in points:
        if 0 <= y < heatmap.shape[0] and 0 <= x < heatmap.shape[1]:
            heatmap[y, x] += 1.0
    
    heatmap = gaussian_filter(heatmap, sigma=15)
    heatmap /= heatmap.max()
    
    out_path = OUT_DIR / mat_path.with_suffix('.png').name
    Image.fromarray((heatmap * 255).astype(np.uint8)).save(out_path)
