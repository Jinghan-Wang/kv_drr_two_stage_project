import argparse
import os
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import SpineDataset
from models import TwoStageNet
from losses import SSIMLoss, GradientLoss
from utils import AverageMeter, seed_everything, ensure_dir, save_validation_panel, write_log



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--img_size", type=int, nargs=2, default=[512, 512])
    parser.add_argument("--max_intensity", type=float, default=4095.0)
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--lambda_drr_l1", type=float, default=1.0)
    parser.add_argument("--lambda_drr_ssim", type=float, default=0.5)
    parser.add_argument("--lambda_drr_grad", type=float, default=0.5)

    parser.add_argument("--lambda_spine_l1", type=float, default=1.0)
    parser.add_argument("--lambda_spine_ssim", type=float, default=0.5)
    parser.add_argument("--lambda_spine_grad", type=float, default=0.5)

    parser.add_argument("--detach_stage1_to_stage2", type=int, default=0,
                        help="1: stage2 loss does not backprop to stage1; 0: full end-to-end")

    parser.add_argument("--save_val_images", type=int, default=4,
                        help="Number of validation samples to save per epoch")
    return parser.parse_args()



def build_dataloaders(args):
    train_set = SpineDataset(
        data_root=args.data_root,
        split="train",
        img_size=tuple(args.img_size),
        max_intensity=args.max_intensity,
    )
    val_set = SpineDataset(
        data_root=args.data_root,
        split="val",
        img_size=tuple(args.img_size),
        max_intensity=args.max_intensity,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader



def train_one_epoch(model, loader, optimizer, l1_loss, ssim_loss, grad_loss, device, args):
    model.train()

    total_meter = AverageMeter()
    drr_meter = AverageMeter()
    spine_meter = AverageMeter()

    pbar = tqdm(loader, desc="Train", leave=False)
    for batch in pbar:
        kv = batch["kv"].to(device)
        drr_gt = batch["drr"][:, 0:1].to(device)
        spine_gt = batch["spine"].to(device)

        outputs = model(kv)
        fake_drr = outputs["fake_drr"]
        pred_spine = outputs["pred_spine"]

        loss_drr_l1 = l1_loss(fake_drr, drr_gt)
        loss_drr_ssim = ssim_loss(fake_drr, drr_gt)
        loss_drr_grad = grad_loss(fake_drr, drr_gt)
        loss_drr = (
            args.lambda_drr_l1 * loss_drr_l1
            + args.lambda_drr_ssim * loss_drr_ssim
            + args.lambda_drr_grad * loss_drr_grad
        )

        loss_spine_l1 = l1_loss(pred_spine, spine_gt)
        loss_spine_ssim = ssim_loss(pred_spine, spine_gt)
        loss_spine_grad = grad_loss(pred_spine, spine_gt)
        loss_spine = (
            args.lambda_spine_l1 * loss_spine_l1
            + args.lambda_spine_ssim * loss_spine_ssim
            + args.lambda_spine_grad * loss_spine_grad
        )

        loss = loss_drr + loss_spine

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        bs = kv.size(0)
        total_meter.update(loss.item(), bs)
        drr_meter.update(loss_drr.item(), bs)
        spine_meter.update(loss_spine.item(), bs)

        pbar.set_postfix({
            "loss": f"{total_meter.avg:.4f}",
            "drr": f"{drr_meter.avg:.4f}",
            "spine": f"{spine_meter.avg:.4f}",
        })

    return total_meter.avg, drr_meter.avg, spine_meter.avg


@torch.no_grad()
def validate(model, loader, l1_loss, ssim_loss, grad_loss, device, args, save_dir, epoch):
    model.eval()

    total_meter = AverageMeter()
    drr_meter = AverageMeter()
    spine_meter = AverageMeter()

    vis_dir = os.path.join(save_dir, "val_images", f"epoch_{epoch:04d}")
    ensure_dir(vis_dir)

    pbar = tqdm(loader, desc="Val", leave=False)
    for i, batch in enumerate(pbar):
        kv = batch["kv"].to(device)
        drr_gt = batch["drr"][:, 0:1].to(device)
        spine_gt = batch["spine"].to(device)
        name = batch["name"][0]

        outputs = model(kv)
        fake_drr = outputs["fake_drr"]
        pred_spine = outputs["pred_spine"]

        loss_drr_l1 = l1_loss(fake_drr, drr_gt)
        loss_drr_ssim = ssim_loss(fake_drr, drr_gt)
        loss_drr_grad = grad_loss(fake_drr, drr_gt)
        loss_drr = (
            args.lambda_drr_l1 * loss_drr_l1
            + args.lambda_drr_ssim * loss_drr_ssim
            + args.lambda_drr_grad * loss_drr_grad
        )

        loss_spine_l1 = l1_loss(pred_spine, spine_gt)
        loss_spine_ssim = ssim_loss(pred_spine, spine_gt)
        loss_spine_grad = grad_loss(pred_spine, spine_gt)
        loss_spine = (
            args.lambda_spine_l1 * loss_spine_l1
            + args.lambda_spine_ssim * loss_spine_ssim
            + args.lambda_spine_grad * loss_spine_grad
        )

        loss = loss_drr + loss_spine

        bs = kv.size(0)
        total_meter.update(loss.item(), bs)
        drr_meter.update(loss_drr.item(), bs)
        spine_meter.update(loss_spine.item(), bs)

        pbar.set_postfix({
            "loss": f"{total_meter.avg:.4f}",
            "drr": f"{drr_meter.avg:.4f}",
            "spine": f"{spine_meter.avg:.4f}",
        })

        if i < args.save_val_images:
            # 只保存原始第一通道的 KV
            save_path = os.path.join(vis_dir, f"{name}.png")
            save_validation_panel(
                save_path=save_path,
                kv=kv[0, 0:1],
                drr_gt=drr_gt[0],
                drr_pred=fake_drr[0],
                spine_gt=spine_gt[0],
                spine_pred=pred_spine[0],
            )

    return total_meter.avg, drr_meter.avg, spine_meter.avg



def main():
    args = parse_args()
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ensure_dir(args.save_dir)
    ckpt_dir = os.path.join(args.save_dir, "checkpoints")
    ensure_dir(ckpt_dir)

    log_path = os.path.join(args.save_dir, "train_log.txt")
    write_log(log_path, f"Start: {datetime.now()}")
    write_log(log_path, str(args))

    train_loader, val_loader = build_dataloaders(args)

    model = TwoStageNet(
        max_intensity=args.max_intensity,
        detach_stage1_to_stage2=bool(args.detach_stage1_to_stage2),
        base_channels=args.base_channels,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    l1_loss = torch.nn.L1Loss()
    ssim_loss = SSIMLoss(channel=1).to(device)
    grad_loss = GradientLoss().to(device)

    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_total, train_drr, train_spine = train_one_epoch(
            model, train_loader, optimizer,
            l1_loss, ssim_loss, grad_loss,
            device, args,
        )

        val_total, val_drr, val_spine = validate(
            model, val_loader,
            l1_loss, ssim_loss, grad_loss,
            device, args,
            args.save_dir, epoch,
        )

        log_line = (
            f"Epoch [{epoch:03d}/{args.epochs:03d}] | "
            f"train_total={train_total:.6f}, train_drr={train_drr:.6f}, train_spine={train_spine:.6f} | "
            f"val_total={val_total:.6f}, val_drr={val_drr:.6f}, val_spine={val_spine:.6f}"
        )
        print(log_line)
        write_log(log_path, log_line)

        latest_path = os.path.join(ckpt_dir, "latest.pth")
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
            "best_val": best_val,
        }, latest_path)

        if val_total < best_val:
            best_val = val_total
            best_path = os.path.join(ckpt_dir, "best.pth")
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "args": vars(args),
                "best_val": best_val,
            }, best_path)
            write_log(log_path, f"Saved best checkpoint at epoch {epoch}, val_total={val_total:.6f}")

    write_log(log_path, f"End: {datetime.now()}")


if __name__ == "__main__":
    main()

