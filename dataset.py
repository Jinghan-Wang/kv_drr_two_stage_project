import os
from glob import glob
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torch.nn.functional as F


IMG_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".npy"]


def is_image_file(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    return ext in IMG_EXTS



def load_grayscale_image(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        arr = np.load(path)
    else:
        img = Image.open(path)
        arr = np.array(img)

    if arr.ndim == 3:
        # 如果误读成 RGB，取第一个通道
        arr = arr[..., 0]
    return arr.astype(np.float32)



def threshold_to_max(img: np.ndarray, threshold: float) -> np.ndarray:
    """保留 [threshold, max]，其余置零。"""
    out = img.copy()
    out[out < threshold] = 0.0
    return out



def normalize_image(img: np.ndarray, max_intensity: float) -> np.ndarray:
    img = img / float(max_intensity)
    img = np.clip(img, 0.0, 1.0)
    return img



def resize_tensor_image(img: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    # img: (1, H, W) or (C, H, W)
    img = img.unsqueeze(0)
    img = F.interpolate(img, size=size, mode="bilinear", align_corners=False)
    img = img.squeeze(0)
    return img


class SpineDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        split: str = "train",
        img_size: Tuple[int, int] = (512, 512),
        max_intensity: float = 4095.0,
    ):
        super().__init__()
        self.data_root = data_root
        self.split = split
        self.img_size = img_size
        self.max_intensity = max_intensity

        self.kv_dir = os.path.join(data_root, split, "kv")
        self.drr_dir = os.path.join(data_root, split, "drr")
        self.spine_dir = os.path.join(data_root, split, "spine")

        if not os.path.isdir(self.kv_dir):
            raise FileNotFoundError(f"KV dir not found: {self.kv_dir}")
        if not os.path.isdir(self.drr_dir):
            raise FileNotFoundError(f"DRR dir not found: {self.drr_dir}")
        if not os.path.isdir(self.spine_dir):
            raise FileNotFoundError(f"Spine dir not found: {self.spine_dir}")

        self.samples = self._collect_samples()
        if len(self.samples) == 0:
            raise RuntimeError(f"No samples found under {data_root}/{split}")

    def _collect_samples(self) -> List[Tuple[str, str, str]]:
        kv_files = []
        for ext in IMG_EXTS:
            kv_files.extend(glob(os.path.join(self.kv_dir, f"*{ext}")))
        kv_files = sorted(kv_files)

        samples = []
        for kv_path in kv_files:
            name = os.path.splitext(os.path.basename(kv_path))[0]
            drr_path = self._find_file(self.drr_dir, name)
            spine_path = self._find_file(self.spine_dir, name)
            if drr_path is not None and spine_path is not None:
                samples.append((kv_path, drr_path, spine_path))
        return samples

    @staticmethod
    def _find_file(folder: str, stem: str):
        for ext in IMG_EXTS:
            p = os.path.join(folder, stem + ext)
            if os.path.exists(p):
                return p
        return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        kv_path, drr_path, spine_path = self.samples[idx]

        kv = load_grayscale_image(kv_path)
        drr = load_grayscale_image(drr_path)
        spine = load_grayscale_image(spine_path)

        # 预处理阈值图
        kv_2000 = threshold_to_max(kv, 2000)
        kv_3000 = threshold_to_max(kv, 3000)
        kv_4000 = threshold_to_max(kv, 4000)

        drr_1000 = threshold_to_max(drr, 1000)
        drr_2000 = threshold_to_max(drr, 2000)
        drr_3000 = threshold_to_max(drr, 3000)

        # 归一化
        kv = normalize_image(kv, self.max_intensity)
        kv_2000 = normalize_image(kv_2000, self.max_intensity)
        kv_3000 = normalize_image(kv_3000, self.max_intensity)
        kv_4000 = normalize_image(kv_4000, self.max_intensity)

        drr = normalize_image(drr, self.max_intensity)
        drr_1000 = normalize_image(drr_1000, self.max_intensity)
        drr_2000 = normalize_image(drr_2000, self.max_intensity)
        drr_3000 = normalize_image(drr_3000, self.max_intensity)

        spine = normalize_image(spine, self.max_intensity)

        kv_4ch = np.stack([kv, kv_2000, kv_3000, kv_4000], axis=0)
        drr_4ch = np.stack([drr, drr_1000, drr_2000, drr_3000], axis=0)
        spine_1ch = np.expand_dims(spine, axis=0)

        kv_4ch = torch.from_numpy(kv_4ch).float()
        drr_4ch = torch.from_numpy(drr_4ch).float()
        spine_1ch = torch.from_numpy(spine_1ch).float()

        kv_4ch = resize_tensor_image(kv_4ch, self.img_size)
        drr_4ch = resize_tensor_image(drr_4ch, self.img_size)
        spine_1ch = resize_tensor_image(spine_1ch, self.img_size)

        sample = {
            "kv": kv_4ch,
            "drr": drr_4ch,
            "spine": spine_1ch,
            "name": os.path.splitext(os.path.basename(kv_path))[0],
        }
        return sample

