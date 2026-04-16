from typing import Iterable, Dict
import torch
import torch.nn as nn

from src.models.unet2d import UNet2D
from src.utils.thresholds import build_multi_threshold_channels


class TotalNet(nn.Module):
    def __init__(self,
                 kv_thresholds: Iterable[float],
                 drr_thresholds: Iterable[float],
                 base_channels: int = 32,
                 norm: str = "batch",
                 act: str = "relu",
                 detach_stage1_to_stage2: bool = True):
        super().__init__()
        self.kv_thresholds = list(kv_thresholds)
        self.drr_thresholds = list(drr_thresholds)
        self.detach_stage1_to_stage2 = detach_stage1_to_stage2

        self.stage1 = UNet2D(in_channels=1 + len(self.kv_thresholds), out_channels=1,
                             base_channels=base_channels, norm=norm, act=act)
        self.stage2 = UNet2D(in_channels=1 + len(self.drr_thresholds), out_channels=1,
                             base_channels=base_channels, norm=norm, act=act)

    def forward(self, kv_4ch: torch.Tensor) -> Dict[str, torch.Tensor]:
        fake_drr = self.stage1(kv_4ch)

        stage2_in = fake_drr.detach() if self.detach_stage1_to_stage2 else fake_drr
        fake_drr_4ch = build_multi_threshold_channels(stage2_in, self.drr_thresholds)
        pred_spine = self.stage2(fake_drr_4ch)

        return {
            "fake_drr": fake_drr,
            "fake_drr_4ch": fake_drr_4ch,
            "pred_spine": pred_spine,
        }
