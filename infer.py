import argparse
import os
from glob import glob

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from models import TwoStageNet
from utils import ensure_dir, save_validation_panel


IMG_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".npy"]



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--img_size", type=int, nargs=2, default=[512, 512])
    parser.add_argument("--max_intensity", type=float, default=4095.0)
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--detach_stage1_to_stage2", type=int, default=0)
    return parser.parse_args()



def load_grayscale_image(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        arr = np.load(path)
    else:
        img = Image.open(path)
        arr = np.array(img)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr.astype(np.float32)



def threshold_to_max(img: np.ndarray, threshold: float) -> np.ndarray:
    out = img.copy()
    out[out < threshold] = 0.0
    return out



def normalize_image(img: np.ndarray, max_intensity: float) -> np.ndarray:
    img = img / float(max_intensity)
    img = np.clip(img, 0.0, 1.0)
    return img



def resize_tensor_image(img: torch.Tensor, size):
    img = img.unsqueeze(0)
    img = F.interpolate(img, size=size, mode="bilinear", align_corners=False)
    return img



def tensor_to_pil(img: torch.Tensor) -> Image.Image:
    img = img.squeeze().detach().cpu().numpy()
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(img, mode="L")



def collect_files(input_dir):
    files = []
    for ext in IMG_EXTS:
        files.extend(glob(os.path.join(input_dir, f"*{ext}")))
    return sorted(files)



def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ensure_dir(args.output_dir)
    ensure_dir(os.path.join(args.output_dir, "fake_drr"))
    ensure_dir(os.path.join(args.output_dir, "pred_spine"))
    ensure_dir(os.path.join(args.output_dir, "panels"))

    model = TwoStageNet(
        max_intensity=args.max_intensity,
        detach_stage1_to_stage2=bool(args.detach_stage1_to_stage2),
        base_channels=args.base_channels,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    files = collect_files(args.input_dir)
    if len(files) == 0:
        raise RuntimeError(f"No test KV files found in: {args.input_dir}")

    with torch.no_grad():
        for path in tqdm(files, desc="Infer"):
            name = os.path.splitext(os.path.basename(path))[0]
            kv = load_grayscale_image(path)

            kv_2000 = threshold_to_max(kv, 2000)
            kv_3000 = threshold_to_max(kv, 3000)
            kv_4000 = threshold_to_max(kv, 4000)

            kv = normalize_image(kv, args.max_intensity)
            kv_2000 = normalize_image(kv_2000, args.max_intensity)
            kv_3000 = normalize_image(kv_3000, args.max_intensity)
            kv_4000 = normalize_image(kv_4000, args.max_intensity)

            kv_4ch = np.stack([kv, kv_2000, kv_3000, kv_4000], axis=0)
            kv_4ch = torch.from_numpy(kv_4ch).float()
            kv_4ch = resize_tensor_image(kv_4ch, tuple(args.img_size)).to(device)

            out = model(kv_4ch)
            fake_drr = out["fake_drr"][0]
            pred_spine = out["pred_spine"][0]
            kv_vis = kv_4ch[0, 0:1]

            tensor_to_pil(fake_drr).save(os.path.join(args.output_dir, "fake_drr", f"{name}.png"))
            tensor_to_pil(pred_spine).save(os.path.join(args.output_dir, "pred_spine", f"{name}.png"))

            # 推理时没有 GT，这里只输出三张图的简单 panel
            # 复用 save_validation_panel 时需要 5 张，所以自己做简版
            kv_img = tensor_to_pil(kv_vis)
            fake_img = tensor_to_pil(fake_drr)
            spine_img = tensor_to_pil(pred_spine)

            w, h = kv_img.size
            pad_h = 22
            from PIL import ImageDraw
            def add_title(img, title):
                out = Image.new("L", (w, h + pad_h), color=255)
                out.paste(img, (0, pad_h))
                draw = ImageDraw.Draw(out)
                draw.text((5, 5), title, fill=0)
                return out

            ims = [add_title(kv_img, "KV"), add_title(fake_img, "fake_DRR"), add_title(spine_img, "pred_spine")]
            canvas = Image.new("L", (w * 3, h + pad_h), color=255)
            canvas.paste(ims[0], (0, 0))
            canvas.paste(ims[1], (w, 0))
            canvas.paste(ims[2], (2 * w, 0))
            canvas.save(os.path.join(args.output_dir, "panels", f"{name}.png"))


if __name__ == "__main__":
    main()

