import random
from pathlib import Path

import numpy as np
import scipy.io
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

from models import config


class DUTOMRONDataset(Dataset):
    def __init__(self, img_transform=None, heatmap_transform=None):
        self.root_dir = Path(config.DUTOMRON_DIR)
        self.img_transform = img_transform
        self.heatmap_transform = heatmap_transform

        self.img_dir = self.root_dir / "DUT-OMRON-image"
        self.img_list = sorted(self.img_dir.glob("*.jpg"))
        self.heatmap_dir = self.root_dir / "DUT-OMRON-heatmaps"

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.img_list[idx]
        heatmap_path = self.heatmap_dir / img_path.with_suffix(".png").name

        img = Image.open(img_path).convert("RGB")
        heatmap = Image.open(heatmap_path).convert("L")

        sample = {"image": img, "heatmap": heatmap}

        if self.img_transform:
            sample["image"] = self.img_transform(sample["image"])
        if self.heatmap_transform:
            sample["heatmap"] = self.heatmap_transform(sample["heatmap"])

        return sample


class SALICONDataset(Dataset):
    def __init__(
        self, split, img_transform=None, heatmap_transform=None, fixation_transform=None
    ):
        self.root_dir = Path(config.SALICON_DIR)
        self.split = split
        self.img_transform = img_transform
        self.heatmap_transform = heatmap_transform
        self.fixation_transform = fixation_transform

        self.img_dir = self.root_dir / "images" / self.split
        self.img_list = sorted(self.img_dir.glob("*.jpg"))
        self.heatmap_dir = self.root_dir / "maps" / self.split
        self.fixation_dir = self.root_dir / "fixations" / self.split

    def load_fixation(self, mat_path):
        data = scipy.io.loadmat(mat_path)
        h, w = (int(v) for v in data["resolution"][0])
        fmap = np.zeros((h, w), dtype=np.uint8)
        gaze = data["gaze"]
        for i in range(gaze.shape[0]):
            fixations = gaze[i, 0]["fixations"]
            if fixations.size == 0:
                continue
            xs = fixations[:, 0].astype(int)
            ys = fixations[:, 1].astype(int)
            valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
            fmap[ys[valid], xs[valid]] = 255
        return Image.fromarray(fmap, mode="L")

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.img_list[idx]
        img = Image.open(img_path).convert("RGB")

        heatmap = None
        fixation = None
        if self.split in ["train", "val"]:
            heatmap_path = self.heatmap_dir / img_path.with_suffix(".png").name
            heatmap = Image.open(heatmap_path).convert("L")
            fixation = self.load_fixation(
                self.fixation_dir / img_path.with_suffix(".mat").name
            )

        if self.split == "train":
            if random.random() < 0.5:
                img = T.functional.hflip(img)
                heatmap = T.functional.hflip(heatmap)
                fixation = T.functional.hflip(fixation)

            if random.random() < 0.5:
                angle = random.uniform(-10, 10)
                img = T.functional.rotate(
                    img, angle, interpolation=T.functional.InterpolationMode.BILINEAR
                )
                heatmap = T.functional.rotate(
                    heatmap, angle, interpolation=T.functional.InterpolationMode.NEAREST
                )
                fixation = T.functional.rotate(
                    fixation,
                    angle,
                    interpolation=T.functional.InterpolationMode.NEAREST,
                )

            if random.random() < 0.5:
                color_jitter = T.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
                )
                img = color_jitter(img)

        sample = {"image": img}
        if heatmap is not None:
            sample["heatmap"] = heatmap
        if fixation is not None:
            sample["fixation"] = fixation

        if self.img_transform:
            sample["image"] = self.img_transform(sample["image"])
        if self.heatmap_transform:
            sample["heatmap"] = self.heatmap_transform(sample["heatmap"])
        if self.fixation_transform and "fixation" in sample:
            sample["fixation"] = self.fixation_transform(sample["fixation"])

        return sample


class VATDDataset(Dataset):
    def __init__(
        self, split, img_transform=None, depth_transform=None, blur_transform=None
    ):
        self.root_dir = Path(config.VATD_DIR)
        self.split = split
        self.img_transform = img_transform
        self.depth_transform = depth_transform
        self.blur_transform = blur_transform

        self.img_dir = self.root_dir / self.split / "input"
        self.img_list = sorted(self.img_dir.glob("*.JPG"))

        self.depth_dir = self.root_dir / self.split / "depth"
        self.blur_dir = self.root_dir / self.split / "2_8"

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.img_list[idx]
        depth_path = self.depth_dir / img_path.with_suffix(".png").name
        blur_path = self.blur_dir / img_path.name

        img = Image.open(img_path).convert("RGB")
        depth = Image.open(depth_path).convert("L")
        blur = Image.open(blur_path).convert("RGB")

        sample = {"image": img, "depth": depth, "blur": blur}

        if self.img_transform:
            sample["image"] = self.img_transform(sample["image"])
        if self.depth_transform:
            sample["depth"] = self.depth_transform(sample["depth"])
        if self.blur_transform:
            sample["blur"] = self.blur_transform(sample["blur"])

        return sample
