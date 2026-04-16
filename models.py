from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch)
        )

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            self.conv = DoubleConv(in_ch, out_ch)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=1, base_channels=32, bilinear=True):
        super().__init__()
        self.inc = DoubleConv(in_channels, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        self.down4 = Down(base_channels * 8, base_channels * 8)

        self.up1 = Up(base_channels * 16, base_channels * 4, bilinear)
        self.up2 = Up(base_channels * 8, base_channels * 2, bilinear)
        self.up3 = Up(base_channels * 4, base_channels, bilinear)
        self.up4 = Up(base_channels * 2, base_channels, bilinear)
        self.outc = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        x = torch.sigmoid(x)
        return x


class TwoStageNet(nn.Module):
    """
    Stage1: KV(4ch) -> fake_DRR(1ch)
    Stage2: fake_DRR + thresholds [1000,2000,3000] -> pred_spine(1ch)
    """
    def __init__(
        self,
        max_intensity: float = 4095.0,
        detach_stage1_to_stage2: bool = False,
        base_channels: int = 32,
    ):
        super().__init__()
        self.max_intensity = max_intensity
        self.detach_stage1_to_stage2 = detach_stage1_to_stage2
        self.stage1 = UNet(in_channels=4, out_channels=1, base_channels=base_channels)
        self.stage2 = UNet(in_channels=4, out_channels=1, base_channels=base_channels)

    def threshold_to_max(self, img01: torch.Tensor, threshold: float) -> torch.Tensor:
        """
        img01: normalized image in [0, 1], shape (B,1,H,W)
        threshold is in original intensity domain.
        """
        thr = threshold / self.max_intensity
        out = img01.clone()
        out = torch.where(out >= thr, out, torch.zeros_like(out))
        return out

    def make_fake_drr_4ch(self, fake_drr: torch.Tensor) -> torch.Tensor:
        d1000 = self.threshold_to_max(fake_drr, 1000)
        d2000 = self.threshold_to_max(fake_drr, 2000)
        d3000 = self.threshold_to_max(fake_drr, 3000)
        return torch.cat([fake_drr, d1000, d2000, d3000], dim=1)

    def forward(self, kv_4ch: torch.Tensor):
        fake_drr = self.stage1(kv_4ch)
        stage2_input = fake_drr.detach() if self.detach_stage1_to_stage2 else fake_drr
        fake_drr_4ch = self.make_fake_drr_4ch(stage2_input)
        pred_spine = self.stage2(fake_drr_4ch)
        return {
            "fake_drr": fake_drr,
            "fake_drr_4ch": fake_drr_4ch,
            "pred_spine": pred_spine,
        }

