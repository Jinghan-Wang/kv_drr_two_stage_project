import csv
from typing import Dict, List
import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils.io import load_nifti_2d
from src.utils.thresholds import build_multi_threshold_channels


class PairedNiftiDataset(Dataset):
    def __init__(self, csv_file: str, intensity_scale: float, kv_thresholds: List[float], drr_thresholds: List[float]):
        self.items = []
        self.intensity_scale = float(intensity_scale)
        self.kv_thresholds = [float(x) / self.intensity_scale for x in kv_thresholds]
        self.drr_thresholds = [float(x) / self.intensity_scale for x in drr_thresholds]

        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.items.append(row)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.items[idx]
        kv, affine, header = load_nifti_2d(item["kv"])
        drr, _, _ = load_nifti_2d(item["drr"])
        spine, _, _ = load_nifti_2d(item["spine"])

        kv = kv.astype(np.float32) / self.intensity_scale
        drr = drr.astype(np.float32) / self.intensity_scale
        spine = spine.astype(np.float32) / self.intensity_scale

        kv_t = torch.from_numpy(kv)
        drr_t = torch.from_numpy(drr)
        spine_t = torch.from_numpy(spine).unsqueeze(0)  # (1, H, W)

        kv_4ch = build_multi_threshold_channels(kv_t, self.kv_thresholds).squeeze(0)   # (4, H, W)
        drr_4ch = build_multi_threshold_channels(drr_t, self.drr_thresholds).squeeze(0) # (4, H, W)

        return {
            "kv_4ch": kv_4ch,
            "drr_4ch": drr_4ch,
            "drr": drr_t.unsqueeze(0),
            "spine": spine_t,
            "kv_path": item["kv"],
            "drr_path": item["drr"],
            "spine_path": item["spine"],
            "affine": torch.from_numpy(affine.astype(np.float32)),
            "header_placeholder": torch.tensor([0.0]),  # header can't be directly collated; kept via kv_path when saving
        }
