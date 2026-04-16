from typing import Tuple
import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, norm: str = "batch", act: str = "relu"):
        super().__init__()
        if norm == "batch":
            norm_layer = nn.BatchNorm2d
        elif norm == "instance":
            norm_layer = nn.InstanceNorm2d
        else:
            raise ValueError(f"Unsupported norm: {norm}")

        if act == "relu":
            act_layer = nn.ReLU
        elif act == "leaky_relu":
            act_layer = lambda inplace=True: nn.LeakyReLU(0.1, inplace=inplace)
        else:
            raise ValueError(f"Unsupported act: {act}")

        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            norm_layer(out_ch),
            act_layer(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            norm_layer(out_ch),
            act_layer(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, norm: str, act: str):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch, norm, act),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Up(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, norm: str, act: str):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch // 2 + skip_ch, out_ch, norm, act)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        x = nn.functional.pad(x, [diff_x // 2, diff_x - diff_x // 2,
                                  diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 1, base_channels: int = 32,
                 norm: str = "batch", act: str = "relu"):
        super().__init__()
        self.inc = DoubleConv(in_channels, base_channels, norm, act)
        self.down1 = Down(base_channels, base_channels * 2, norm, act)
        self.down2 = Down(base_channels * 2, base_channels * 4, norm, act)
        self.down3 = Down(base_channels * 4, base_channels * 8, norm, act)
        self.down4 = Down(base_channels * 8, base_channels * 16, norm, act)
        self.up1 = Up(base_channels * 16, base_channels * 8, base_channels * 8, norm, act)
        self.up2 = Up(base_channels * 8, base_channels * 4, base_channels * 4, norm, act)
        self.up3 = Up(base_channels * 4, base_channels * 2, base_channels * 2, norm, act)
        self.up4 = Up(base_channels * 2, base_channels, base_channels, norm, act)
        self.outc = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        return x
