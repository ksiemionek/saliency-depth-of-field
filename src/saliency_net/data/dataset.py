from torch.utils.data import Dataset
import torch
from PIL import Image
from pathlib import Path


class DUTOMRONDataset(Dataset):
    def __init__(self, transform=None):
        self.img_dir = Path('./DUT-OMRON/DUT-OMRON-image')
        self.img_list = sorted(self.img_dir.glob('*.jpg'))
        self.heatmap_dir = Path('./DUT-OMRON/DUT-OMRON-heatmaps')
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_path = self.img_list[idx]
        heatmap_path = self.heatmap_dir / img_path.with_suffix('.png').name

        img = Image.open(img_path).convert('RGB')
        heatmap = Image.open(heatmap_path).convert('L')
        
        sample = {'image': img, 'heatmap': heatmap}

        if self.transform:
            sample['image'] = self.transform(sample['image'])
            sample['heatmap'] = self.transform(sample['heatmap'])
        
        return sample
