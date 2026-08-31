"""Stage 12: resolution-robust, mask-aware spectral recovery model.

The model is designed to accept variable-length, irregularly sampled spectra. It
uses wavelength coordinates and local wavelength spacing explicitly, plus the
observed uncertainty and a wavelength-aware classical baseline. It does not
interpolate missing regions or require a fixed wavelength grid.
"""
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        p = dilation
        self.c1 = nn.Conv1d(channels, channels, 3, padding=p, dilation=dilation)
        self.n1 = nn.GroupNorm(8, channels)
        self.c2 = nn.Conv1d(channels, channels, 3, padding=p, dilation=dilation)
        self.n2 = nn.GroupNorm(8, channels)
    def forward(self, x):
        h = F.gelu(self.n1(self.c1(x)))
        h = self.n2(self.c2(h))
        return F.gelu(x + h)

class GeneralizedSpectralRecoveryNet(nn.Module):
    """Variable-length spectral denoiser.

    Channels: absolute normalized wavelength, local spacing, noisy flux,
    uncertainty, wavelength-aware baseline, observed-mask.
    """
    def __init__(self, channels: int = 40):
        super().__init__()
        self.input = nn.Conv1d(6, channels, 7, padding=3)
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
        sigma = x[:, 3:4].clamp_min(1e-4)
        baseline = x[:, 4:5]
        mask = x[:, 5:6]
        correction = 2.0 * sigma * torch.tanh(out[:, :1])
        mean = baseline + correction
        mean = mean * mask + baseline * (1 - mask)
        log_sigma = torch.clamp(torch.log(sigma) + 0.55 * torch.tanh(out[:, 1:2]), -8.0, -1.0)
        return mean, log_sigma

def masked_gaussian_nll(mean, log_sigma, target, mask):
    sigma = torch.exp(log_sigma).clamp_min(1e-4)
    nll = 0.5 * ((target - mean) / sigma) ** 2 + log_sigma
    return (nll * mask).sum() / mask.sum().clamp_min(1.0)

def _pair_mask(mask):
    return mask[..., 1:] * mask[..., :-1]

def _triple_mask(mask):
    return mask[..., 2:] * mask[..., 1:-1] * mask[..., :-2]

def loss_fn(mean, log_sigma, target, mask):
    nll = masked_gaussian_nll(mean, log_sigma, target, mask)
    mse = (((mean-target)**2)*mask).sum()/mask.sum().clamp_min(1.0)
    gd = mean[...,1:]-mean[...,:-1]
    gt = target[...,1:]-target[...,:-1]
    pm = _pair_mask(mask)
    grad = (((gd-gt)**2)*pm).sum()/pm.sum().clamp_min(1.0)
    cd = mean[...,2:]-2*mean[...,1:-1]+mean[...,:-2]
    ct = target[...,2:]-2*target[...,1:-1]+target[...,:-2]
    tm = _triple_mask(mask)
    curv = (((cd-ct)**2)*tm).sum()/tm.sum().clamp_min(1.0)
    return nll + 0.30*mse + 0.16*grad + 0.05*curv, {"nll":nll.detach(),"mse":mse.detach(),"gradient":grad.detach(),"curvature":curv.detach()}
