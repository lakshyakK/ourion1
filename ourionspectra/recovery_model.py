"""Structure-preserving, uncertainty-aware 1-D spectral recovery network.

Stage 7 adds a model-internal classical-denoising reference (masked local
baseline) as a feature. The network learns a noise-conditioned correction to
that baseline rather than being forced to reconstruct from raw noisy flux alone.
This is a hybrid denoising model, not a replacement for the scientific
reference spectrum and not an atmospheric-feature detector.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        pad = dilation
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=pad, dilation=dilation)
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=pad, dilation=dilation)
        self.norm2 = nn.GroupNorm(8, channels)

    def forward(self, x):
        h = F.gelu(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return F.gelu(x + h)


class SpectralRecoveryNet(nn.Module):
    """Noise-aware 1-D CNN using raw flux, sigma and a classical baseline.

    Channels: normalized wavelength, noisy flux, empirical sigma, masked
    moving-average baseline. The prediction is a bounded correction to the
    baseline, making the model explicitly competitive with a transparent
    classical reference rather than silently replacing it.
    """

    def __init__(self, channels: int = 32):
        super().__init__()
        self.input = nn.Conv1d(4, channels, 7, padding=3)
        self.blocks = nn.Sequential(
            ResidualBlock(channels, 1), ResidualBlock(channels, 2),
            ResidualBlock(channels, 4), ResidualBlock(channels, 8),
            ResidualBlock(channels, 16),
        )
        self.head = nn.Conv1d(channels, 2, 5, padding=2)

    def forward(self, x):
        h = F.gelu(self.input(x))
        h = self.blocks(h)
        out = self.head(h)
        sigma = x[:, 2:3, :].clamp_min(1e-4)
        baseline = x[:, 3:4, :]
        correction = 2.5 * sigma * torch.tanh(out[:, :1, :])
        mean = baseline + correction
        log_sigma = torch.clamp(torch.log(sigma) + 0.65 * torch.tanh(out[:, 1:2, :]), -7.0, -1.0)
        return mean, log_sigma


def masked_gaussian_nll(mean, log_sigma, target, mask):
    sigma = torch.exp(log_sigma).clamp_min(1e-4)
    nll = 0.5 * ((target - mean) / sigma) ** 2 + log_sigma
    return (nll * mask).sum() / mask.sum().clamp_min(1.0)


def masked_mse(mean, target, mask):
    return (((mean - target) ** 2) * mask).sum() / mask.sum().clamp_min(1.0)


def masked_gradient_loss(mean, target, mask):
    pred_d = mean[..., 1:] - mean[..., :-1]
    true_d = target[..., 1:] - target[..., :-1]
    pair_mask = mask[..., 1:] * mask[..., :-1]
    return (((pred_d - true_d) ** 2) * pair_mask).sum() / pair_mask.sum().clamp_min(1.0)


def masked_curvature_loss(mean, target, mask):
    pred_c = mean[..., 2:] - 2 * mean[..., 1:-1] + mean[..., :-2]
    true_c = target[..., 2:] - 2 * target[..., 1:-1] + target[..., :-2]
    triplet = mask[..., 2:] * mask[..., 1:-1] * mask[..., :-2]
    return (((pred_c - true_c) ** 2) * triplet).sum() / triplet.sum().clamp_min(1.0)


def recovery_loss(mean, log_sigma, target, mask):
    nll = masked_gaussian_nll(mean, log_sigma, target, mask)
    mse = masked_mse(mean, target, mask)
    grad = masked_gradient_loss(mean, target, mask)
    curv = masked_curvature_loss(mean, target, mask)
    total = nll + 0.35 * mse + 0.20 * grad + 0.08 * curv
    return total, {"nll": nll.detach(), "mse": mse.detach(), "gradient": grad.detach(), "curvature": curv.detach()}
