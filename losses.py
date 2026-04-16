import torch
import torch.nn as nn
import torch.nn.functional as F


class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, channel=1, size_average=True):
        super().__init__()
        self.window_size = window_size
        self.channel = channel
        self.size_average = size_average
        self.register_buffer('window', self.create_window(window_size, channel))

    def gaussian(self, window_size, sigma):
        gauss = torch.tensor([
            torch.exp(torch.tensor(-((x - window_size // 2) ** 2) / float(2 * sigma ** 2)))
            for x in range(window_size)
        ])
        return gauss / gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window @ _1D_window.t()
        window = _2D_window.float().unsqueeze(0).unsqueeze(0)
        window = window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def forward(self, img1, img2):
        _, channel, _, _ = img1.size()
        if channel != self.channel:
            window = self.create_window(self.window_size, channel).to(img1.device)
        else:
            window = self.window.to(img1.device)

        mu1 = F.conv2d(img1, window, padding=self.window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=self.window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=self.window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size // 2, groups=channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        )

        if self.size_average:
            return 1.0 - ssim_map.mean()
        else:
            return 1.0 - ssim_map.mean(1).mean(1).mean(1)


class GradientLoss(nn.Module):
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def gradient(self, x):
        gx = F.conv2d(x, self.sobel_x, padding=1)
        gy = F.conv2d(x, self.sobel_y, padding=1)
        return gx, gy

    def forward(self, pred, target):
        pred_gx, pred_gy = self.gradient(pred)
        targ_gx, targ_gy = self.gradient(target)
        return F.l1_loss(pred_gx, targ_gx) + F.l1_loss(pred_gy, targ_gy)

