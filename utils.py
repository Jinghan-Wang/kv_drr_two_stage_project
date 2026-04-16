import os
import random
from typing import Dict, List

import numpy as np
import torch
from PIL import Image, ImageDraw


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    @property
    def avg(self):
        return self.sum / max(self.count, 1)

    def update(self, val, n=1):
        self.sum += float(val) * n
        self.count += n



def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)



def tensor_to_uint8(img: torch.Tensor) -> np.ndarray:
    """img: (1,H,W) or (H,W) in [0,1]"""
    if img.ndim == 3:
        img = img.squeeze(0)
    img = img.detach().cpu().numpy()
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return img



def add_title_to_image(img: Image.Image, title: str, pad_h: int = 22) -> Image.Image:
    w, h = img.size
    out = Image.new("L", (w, h + pad_h), color=255)
    out.paste(img, (0, pad_h))
    draw = ImageDraw.Draw(out)
    draw.text((5, 5), title, fill=0)
    return out



def save_validation_panel(
    save_path: str,
    kv: torch.Tensor,
    drr_gt: torch.Tensor,
    drr_pred: torch.Tensor,
    spine_gt: torch.Tensor,
    spine_pred: torch.Tensor,
):
    imgs = [
        (kv, "KV"),
        (drr_gt, "DRR_gt"),
        (drr_pred, "fake_DRR"),
        (spine_gt, "spine_gt"),
        (spine_pred, "spine_pred"),
    ]

    pil_list = []
    for tensor, title in imgs:
        arr = tensor_to_uint8(tensor)
        pil = Image.fromarray(arr, mode="L")
        pil = add_title_to_image(pil, title)
        pil_list.append(pil)

    widths, heights = zip(*(im.size for im in pil_list))
    total_w = sum(widths)
    max_h = max(heights)

    canvas = Image.new("L", (total_w, max_h), color=255)
    x = 0
    for im in pil_list:
        canvas.paste(im, (x, 0))
        x += im.size[0]

    canvas.save(save_path)



def write_log(log_path: str, text: str):
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(text + "\n")

