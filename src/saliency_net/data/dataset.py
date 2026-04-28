from torch.utils.data import Dataset
import torch
from PIL import Image
from pathlib import Path


class DUTOMRONDataset(Dataset):
    def __init__(self, img_transform=None, heatmap_transform=None):
        self.root_dir = Path('./data/DUT-OMRON')
        self.img_transform = img_transform
        self.heatmap_transform = heatmap_transform

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

        if self.img_transform:
            sample['image'] = self.img_transform(sample['image'])
        if self.heatmap_transform:
            sample['heatmap'] = self.heatmap_transform(sample['heatmap'])
        
        return sample


class SALICONDataset(Dataset):
    def __init__(self, split, img_transform=None, heatmap_transform=None):
        self.root_dir = Path('./data/SALICON')
        self.split = split
        self.img_transform = img_transform
        self.heatmap_transform = heatmap_transform

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
        
        if self.img_transform:
            sample['image'] = self.img_transform(sample['image'])
        if self.heatmap_transform:
            sample['heatmap'] = self.heatmap_transform(sample['heatmap'])
        
        return sample
