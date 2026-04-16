from typing import Iterable, List
import torch


def threshold_to_max(x: torch.Tensor, thr: float) -> torch.Tensor:
    """
    Keep values >= thr and set below-threshold values to 0.
    x: (..., H, W)
    """
    return torch.where(x >= thr, x, torch.zeros_like(x))


def build_multi_threshold_channels(
    x: torch.Tensor,
    thresholds: Iterable[float],
) -> torch.Tensor:
    """
    Build [original, thr1-max, thr2-max, thr3-max, ...]
    x: (B, 1, H, W) or (1, H, W) or (H, W)
    Return: (B, C, H, W)
    """
    if x.dim() == 2:
        x = x.unsqueeze(0).unsqueeze(0)
    elif x.dim() == 3:
        x = x.unsqueeze(1)

    channels: List[torch.Tensor] = [x]
    for thr in thresholds:
        channels.append(threshold_to_max(x, thr))
    return torch.cat(channels, dim=1)
