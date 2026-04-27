from torch.utils.data import Dataset
import torch
from PIL import Image
from pathlib import Path


class DUTOMRONDataset(Dataset):
    def __init__(self, transform=None):
        self.root_dir = Path('./DUT-OMRON')
        self.transform = transform

        self.img_dir = self.root_dir / 'DUT-OMRON-image'
        self.img_list = sorted(self.img_dir.glob('*.jpg'))
        self.heatmap_dir = self.root_dir / 'DUT-OMRON-heatmaps'
        

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


class SALICONDataset(Dataset):
    def __init__(self, split='train', transform=None):
        self.root_dir = Path('./SALICON')
        self.split = split
        self.transform = transform

        self.img_dir = self.root_dir / 'images' / self.split
        self.img_list = sorted(self.img_dir.glob('*.jpg'))
        self.heatmap_dir = self.root_dir / 'maps' / self.split
        

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_path = self.img_list[idx]
        img = Image.open(img_path).convert('RGB')

        sample = {'image': img}

        if self.split in ['train', 'val']:
            heatmap_path = self.heatmap_dir / img_path.with_suffix('.png').name
            heatmap = Image.open(heatmap_path).convert('L')
            sample['heatmap'] = heatmap
        
        if self.transform:
            sample['image'] = self.transform(sample['image'])
            if self.split in ['train', 'val']:
                sample['heatmap'] = self.transform(sample['heatmap'])
        
        return sample
