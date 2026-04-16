import os
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import PairedNiftiDataset
from src.models.total_net import TotalNet
from src.utils.io import save_nifti_2d, load_nifti_2d
from src.utils.misc import load_config, ensure_dir


def prepare_device(device_str: str) -> torch.device:
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = prepare_device(cfg["train"]["device"])

    ds = PairedNiftiDataset(
        csv_file=cfg["data"]["val_csv"],
        intensity_scale=cfg["data"]["intensity_scale"],
        kv_thresholds=cfg["data"]["kv_thresholds"],
        drr_thresholds=cfg["data"]["drr_thresholds"],
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False,
                        num_workers=cfg["data"]["num_workers"],
                        pin_memory=cfg["data"]["pin_memory"])

    kv_thresholds_scaled = [t / cfg["data"]["intensity_scale"] for t in cfg["data"]["kv_thresholds"]]
    drr_thresholds_scaled = [t / cfg["data"]["intensity_scale"] for t in cfg["data"]["drr_thresholds"]]

    model = TotalNet(
        kv_thresholds=kv_thresholds_scaled,
        drr_thresholds=drr_thresholds_scaled,
        base_channels=cfg["model"]["base_channels"],
        norm=cfg["model"]["norm"],
        act=cfg["model"]["act"],
        detach_stage1_to_stage2=cfg["train"]["detach_stage1_to_stage2"],
    ).to(device)

    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model_state"])
    model.eval()

    output_dir = os.path.join(cfg["output"]["root"], cfg["experiment_name"], "validate_only")
    ensure_dir(output_dir)
    scale = cfg["data"]["intensity_scale"]

    for batch in tqdm(loader, desc="Validate"):
        kv_4ch = batch["kv_4ch"].to(device)
        outputs = model(kv_4ch)

        kv_path = batch["kv_path"][0]
        drr_path = batch["drr_path"][0]
        spine_path = batch["spine_path"][0]
        kv_img, affine, header = load_nifti_2d(kv_path)
        drr_img, _, _ = load_nifti_2d(drr_path)
        spine_img, _, _ = load_nifti_2d(spine_path)

        base = os.path.splitext(os.path.basename(kv_path))[0].replace(".nii", "")
        fake_drr_np = outputs["fake_drr"][0, 0].cpu().numpy() * scale
        pred_spine_np = outputs["pred_spine"][0, 0].cpu().numpy() * scale

        save_nifti_2d(fake_drr_np, affine, header, os.path.join(output_dir, f"{base}_fake_drr.nii.gz"))
        save_nifti_2d(pred_spine_np, affine, header, os.path.join(output_dir, f"{base}_pred_spine.nii.gz"))
        if cfg["val"]["save_inputs"]:
            save_nifti_2d(kv_img, affine, header, os.path.join(output_dir, f"{base}_kv.nii.gz"))
            save_nifti_2d(drr_img, affine, header, os.path.join(output_dir, f"{base}_drr.nii.gz"))
            save_nifti_2d(spine_img, affine, header, os.path.join(output_dir, f"{base}_spine.nii.gz"))

    print(f"Saved validation outputs to {output_dir}")


if __name__ == "__main__":
    main()
