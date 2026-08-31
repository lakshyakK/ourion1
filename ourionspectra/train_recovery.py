"""Train OurionSpectra's uncertainty-aware hybrid spectral recovery network."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .recovery_model import SpectralRecoveryNet, recovery_loss

DEFAULT_DATA = Path("data/wasp39b/training")
DEFAULT_OUTPUT = Path("artifacts/recovery_model")


def masked_moving_average(values, valid, radius=5):
    values = np.asarray(values, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    width = 2 * radius + 1
    sums = np.convolve(np.where(valid, values, 0.0), np.ones(width, dtype=np.float32), mode="same")
    counts = np.convolve(valid.astype(np.float32), np.ones(width, dtype=np.float32), mode="same")
    out = np.full(values.shape, np.nan, dtype=np.float32)
    good = valid & (counts > 0)
    out[good] = sums[good] / counts[good]
    return out


class SpectrumDataset(Dataset):
    def __init__(self, path: str | Path):
        self.samples = json.loads(Path(path).read_text(encoding="utf-8"))
        if not self.samples:
            raise ValueError(f"Empty dataset: {path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        s = self.samples[index]
        wl = np.asarray(s["wavelength"], dtype=np.float32)
        noisy = np.asarray(s["noisy_flux"], dtype=np.float32)
        clean = np.asarray(s["clean_flux"], dtype=np.float32)
        sigma = np.asarray(s["noise_sigma"], dtype=np.float32)
        valid = np.isfinite(clean) & np.isfinite(noisy) & np.isfinite(sigma)
        wl = (wl - 0.63) / (5.17 - 0.63)
        noisy_in = np.nan_to_num(noisy, nan=0.0)
        sigma_in = np.nan_to_num(sigma, nan=0.0)
        baseline = masked_moving_average(noisy, valid, radius=5)
        baseline = np.nan_to_num(baseline, nan=0.0)
        x = np.stack([wl, noisy_in, sigma_in, baseline], axis=0)
        return (
            torch.from_numpy(x),
            torch.from_numpy(np.nan_to_num(clean, nan=0.0)[None, :]),
            torch.from_numpy(valid.astype(np.float32)[None, :]),
        )


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _apply_training_domain_augmentation(x, mask, rng):
    if rng.random() >= 0.60:
        return x
    valid = mask[:, 0, :] > 0
    n = x.shape[-1]
    step = 1.0 / max(n - 1, 1)
    length = float(rng.uniform(0.04, 0.20)) / 4.54
    radius = max(1, int(np.ceil(4 * length / step)))
    grid = torch.arange(-radius, radius + 1, dtype=x.dtype, device=x.device) * step
    kernel = torch.exp(-0.5 * (grid / max(length, 1e-6)) ** 2)
    kernel = kernel / kernel.sum()
    for b in range(x.shape[0]):
        white = torch.randn(n, device=x.device, dtype=x.dtype)
        field = torch.conv1d(white.view(1, 1, -1), kernel.view(1, 1, -1), padding=radius).view(-1)
        field = (field - field.mean()) / field.std().clamp_min(1e-6)
        amplitude = float(rng.uniform(0.05, 0.20))
        x[b, 1, valid[b]] += amplitude * x[b, 2, valid[b]] * field[valid[b]]
        # Recompute the classical feature after the nuisance perturbation.
        vals = x[b, 1].detach().cpu().numpy()
        vm = valid[b].detach().cpu().numpy()
        base = masked_moving_average(vals, vm, radius=5)
        x[b, 3] = torch.from_numpy(np.nan_to_num(base, nan=0.0)).to(x.device, x.dtype)
    return x


def run_epoch(model, loader, optimizer=None, device="cpu", augmentation_rng=None):
    training = optimizer is not None
    model.train(training)
    total, count = 0.0, 0
    for x, y, mask in loader:
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        if training and augmentation_rng is not None:
            x = _apply_training_domain_augmentation(x, mask, augmentation_rng)
        mean, log_sigma = model(x)
        loss, _ = recovery_loss(mean, log_sigma, y, mask)
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total += float(loss.item()) * int(mask.sum().item())
        count += int(mask.sum().item())
    return total / max(count, 1)


def train(data_dir=DEFAULT_DATA, output_dir=DEFAULT_OUTPUT, epochs=30, batch_size=32, learning_rate=2e-3, seed=42):
    set_seed(seed)
    torch.set_num_threads(2)
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_ds = SpectrumDataset(data_dir / "train.json")
    val_ds = SpectrumDataset(data_dir / "val.json")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    model = SpectralRecoveryNet(channels=32).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    augmentation_rng = np.random.default_rng(seed + 1001)
    history, best_val, best_epoch = [], float("inf"), 0
    best_path = output_dir / "model.pt"
    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, device, augmentation_rng)
        with torch.no_grad():
            val_loss = run_epoch(model, val_loader, None, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": val_loss})
        if val_loss < best_val:
            best_val, best_epoch = val_loss, epoch
            torch.save({"model_state": model.state_dict(), "seed": seed, "stage": 7}, best_path)
    report = {
        "model": "SpectralRecoveryNet_hybrid_baseline_conditioned",
        "objective": "masked heteroskedastic Gaussian NLL + flux/gradient/curvature losses",
        "inputs": ["normalized_wavelength", "noisy_flux", "noise_sigma", "masked_moving_average_baseline"],
        "outputs": ["recovered_flux", "predictive_sigma"],
        "device": device, "train_samples": len(train_ds), "validation_samples": len(val_ds),
        "epochs": epochs, "best_epoch": best_epoch, "best_validation_loss": best_val,
        "seed": seed, "test_set_used_for_training": False, "history": history,
        "stage": "7_high_noise_and_uncertainty",
        "scientific_caveat": "All synthetic targets share one WASP-39b reference candidate; results remain a denoising proof of concept, not cross-planet generalization.",
    }
    (output_dir / "stage7_training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(DEFAULT_DATA)); p.add_argument("--output", default=str(DEFAULT_OUTPUT))
    p.add_argument("--epochs", type=int, default=30); p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-3); p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(); print(json.dumps(train(args.data, args.output, args.epochs, args.batch_size, args.lr, args.seed), indent=2))
