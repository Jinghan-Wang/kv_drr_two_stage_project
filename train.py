import os
import time
import argparse
from typing import Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import PairedNiftiDataset
from src.models.total_net import TotalNet
from src.utils.io import save_nifti_2d, load_nifti_2d
from src.utils.losses import SSIMLoss
from src.utils.misc import set_seed, ensure_dir, load_config


def prepare_device(device_str: str) -> torch.device:
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def compute_losses(outputs: Dict[str, torch.Tensor],
                   drr_gt: torch.Tensor,
                   spine_gt: torch.Tensor,
                   ssim_loss,
                   cfg) -> Dict[str, torch.Tensor]:
    fake_drr = outputs["fake_drr"]
    pred_spine = outputs["pred_spine"]

    l1_stage1 = torch.nn.functional.l1_loss(fake_drr, drr_gt)
    ssim_stage1 = ssim_loss(fake_drr, drr_gt)
    loss_stage1 = cfg["train"]["lambda_stage1_l1"] * l1_stage1 + cfg["train"]["lambda_stage1_ssim"] * ssim_stage1

    l1_stage2 = torch.nn.functional.l1_loss(pred_spine, spine_gt)
    ssim_stage2 = ssim_loss(pred_spine, spine_gt)
    loss_stage2 = cfg["train"]["lambda_stage2_l1"] * l1_stage2 + cfg["train"]["lambda_stage2_ssim"] * ssim_stage2

    total = loss_stage1 + loss_stage2
    return {
        "loss_stage1": loss_stage1,
        "loss_stage2": loss_stage2,
        "l1_stage1": l1_stage1,
        "ssim_stage1": ssim_stage1,
        "l1_stage2": l1_stage2,
        "ssim_stage2": ssim_stage2,
        "total": total,
    }


@torch.no_grad()
def validate(model, loader, device, cfg, ssim_loss, out_dir, epoch):
    model.eval()
    val_dir = os.path.join(out_dir, f"val_epoch_{epoch:03d}")
    if (epoch % cfg["val"]["save_every"]) == 0:
        ensure_dir(val_dir)

    total_loss = 0.0
    total_stage1 = 0.0
    total_stage2 = 0.0
    save_count = 0

    for batch in tqdm(loader, desc=f"Validate {epoch}", leave=False):
        kv_4ch = batch["kv_4ch"].to(device)
        drr = batch["drr"].to(device)
        spine = batch["spine"].to(device)

        outputs = model(kv_4ch)
        losses = compute_losses(outputs, drr, spine, ssim_loss, cfg)

        total_loss += losses["total"].item()
        total_stage1 += losses["loss_stage1"].item()
        total_stage2 += losses["loss_stage2"].item()

        if (epoch % cfg["val"]["save_every"]) == 0 and save_count < cfg["val"]["max_save_cases"]:
            for b in range(kv_4ch.size(0)):
                if save_count >= cfg["val"]["max_save_cases"]:
                    break

                kv_path = batch["kv_path"][b]
                drr_path = batch["drr_path"][b]
                spine_path = batch["spine_path"][b]
                kv_img, affine, header = load_nifti_2d(kv_path)
                drr_img, _, _ = load_nifti_2d(drr_path)
                spine_img, _, _ = load_nifti_2d(spine_path)

                scale = cfg["data"]["intensity_scale"]
                fake_drr_np = outputs["fake_drr"][b, 0].detach().cpu().numpy() * scale
                pred_spine_np = outputs["pred_spine"][b, 0].detach().cpu().numpy() * scale

                base = os.path.splitext(os.path.basename(kv_path))[0].replace(".nii", "")
                save_nifti_2d(fake_drr_np, affine, header, os.path.join(val_dir, f"{base}_fake_drr.nii.gz"))
                save_nifti_2d(pred_spine_np, affine, header, os.path.join(val_dir, f"{base}_pred_spine.nii.gz"))

                if cfg["val"]["save_inputs"]:
                    save_nifti_2d(kv_img, affine, header, os.path.join(val_dir, f"{base}_kv.nii.gz"))
                    save_nifti_2d(drr_img, affine, header, os.path.join(val_dir, f"{base}_drr.nii.gz"))
                    save_nifti_2d(spine_img, affine, header, os.path.join(val_dir, f"{base}_spine.nii.gz"))

                save_count += 1

    n = max(1, len(loader))
    return {
        "val_total_loss": total_loss / n,
        "val_stage1_loss": total_stage1 / n,
        "val_stage2_loss": total_stage2 / n,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    device = prepare_device(cfg["train"]["device"])
    output_dir = os.path.join(cfg["output"]["root"], cfg["experiment_name"])
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    ensure_dir(output_dir)
    ensure_dir(ckpt_dir)

    log_path = os.path.join(output_dir, "train.log")
    with open(log_path, "a", encoding="utf-8") as logf:
        logf.write(f"Start training at {time.ctime()}\n")
        logf.write(f"Config: {cfg}\n")

    train_ds = PairedNiftiDataset(
        csv_file=cfg["data"]["train_csv"],
        intensity_scale=cfg["data"]["intensity_scale"],
        kv_thresholds=cfg["data"]["kv_thresholds"],
        drr_thresholds=cfg["data"]["drr_thresholds"],
    )
    val_ds = PairedNiftiDataset(
        csv_file=cfg["data"]["val_csv"],
        intensity_scale=cfg["data"]["intensity_scale"],
        kv_thresholds=cfg["data"]["kv_thresholds"],
        drr_thresholds=cfg["data"]["drr_thresholds"],
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=cfg["data"]["pin_memory"],
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=cfg["data"]["pin_memory"],
    )

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

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    ssim_loss = SSIMLoss()

    best_metric = float("inf")
    best_name = cfg["train"]["save_best_by"]

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        running_total = 0.0
        running_stage1 = 0.0
        running_stage2 = 0.0

        pbar = tqdm(enumerate(train_loader, start=1), total=len(train_loader), desc=f"Train {epoch}")
        for step, batch in pbar:
            kv_4ch = batch["kv_4ch"].to(device)
            drr = batch["drr"].to(device)
            spine = batch["spine"].to(device)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(kv_4ch)
            losses = compute_losses(outputs, drr, spine, ssim_loss, cfg)
            losses["total"].backward()
            optimizer.step()

            running_total += losses["total"].item()
            running_stage1 += losses["loss_stage1"].item()
            running_stage2 += losses["loss_stage2"].item()

            if step % cfg["train"]["print_freq"] == 0 or step == len(train_loader):
                msg = (
                    f"Epoch [{epoch}/{cfg['train']['epochs']}] Step [{step}/{len(train_loader)}] "
                    f"total={losses['total'].item():.6f} "
                    f"stage1={losses['loss_stage1'].item():.6f} "
                    f"(L1={losses['l1_stage1'].item():.6f}, SSIM={losses['ssim_stage1'].item():.6f}) "
                    f"stage2={losses['loss_stage2'].item():.6f} "
                    f"(L1={losses['l1_stage2'].item():.6f}, SSIM={losses['ssim_stage2'].item():.6f})"
                )
                pbar.set_postfix_str(msg)
                print(msg)
                with open(log_path, "a", encoding="utf-8") as logf:
                    logf.write(msg + "\n")

        train_total = running_total / max(1, len(train_loader))
        train_stage1 = running_stage1 / max(1, len(train_loader))
        train_stage2 = running_stage2 / max(1, len(train_loader))

        summary = f"Epoch {epoch} TRAIN: total={train_total:.6f}, stage1={train_stage1:.6f}, stage2={train_stage2:.6f}"
        print(summary)
        with open(log_path, "a", encoding="utf-8") as logf:
            logf.write(summary + "\n")

        if epoch % cfg["val"]["run_every"] == 0:
            val_metrics = validate(model, val_loader, device, cfg, ssim_loss, output_dir, epoch)
            val_summary = (
                f"Epoch {epoch} VAL: total={val_metrics['val_total_loss']:.6f}, "
                f"stage1={val_metrics['val_stage1_loss']:.6f}, stage2={val_metrics['val_stage2_loss']:.6f}"
            )
            print(val_summary)
            with open(log_path, "a", encoding="utf-8") as logf:
                logf.write(val_summary + "\n")

            metric = val_metrics[best_name]
            state = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "config": cfg,
                **val_metrics,
            }
            torch.save(state, os.path.join(ckpt_dir, "last.pt"))
            if metric < best_metric:
                best_metric = metric
                torch.save(state, os.path.join(ckpt_dir, "best.pt"))
                with open(log_path, "a", encoding="utf-8") as logf:
                    logf.write(f"Saved best checkpoint at epoch {epoch}, {best_name}={metric:.6f}\n")

    print("Training completed.")


if __name__ == "__main__":
    main()
