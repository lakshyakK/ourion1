"""Held-out evaluation for the OurionSpectra recovery network."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .recovery_model import SpectralRecoveryNet, masked_gaussian_nll
from .train_recovery import SpectrumDataset


def evaluate(model_path, test_path, output_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SpectralRecoveryNet().to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    ds = SpectrumDataset(test_path)
    loader = DataLoader(ds, batch_size=32, shuffle=False)

    sqerr = 0.0
    abserr = 0.0
    baseline_sqerr = 0.0
    baseline_abserr = 0.0
    n = 0
    nll_total = 0.0
    coverage = 0
    by_scale = {}
    cursor = 0
    with torch.no_grad():
        for x, y, mask in loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            mean, log_sigma = model(x)
            sigma = torch.exp(log_sigma).clamp_min(1e-4)
            valid = mask.bool()
            err = (mean - y)[valid]
            baseline_err = (x[:, 1:2, :] - y)[valid]
            sig = sigma[valid]
            sqerr += float((err ** 2).sum())
            abserr += float(err.abs().sum())
            baseline_sqerr += float((baseline_err ** 2).sum())
            baseline_abserr += float(baseline_err.abs().sum())
            n += int(valid.sum())
            nll_total += float(masked_gaussian_nll(mean, log_sigma, y, mask).item()) * int(valid.sum())
            coverage += int((err.abs() <= sig).sum())
            for j in range(x.shape[0]):
                sample = ds.samples[cursor + j]
                scale = float(sample["noise_scale"])
                key = "low_held_out" if scale < 0.45 else "high_held_out"
                if key not in by_scale:
                    by_scale[key] = [0.0, 0.0, 0]
                e = (mean[j, 0] - y[j, 0])[valid[j, 0]]
                by_scale[key][0] += float((e ** 2).sum())
                by_scale[key][1] += float(e.abs().sum())
                by_scale[key][2] += int(valid[j, 0].sum())
            cursor += x.shape[0]

    report = {
        "test_samples": len(ds),
        "valid_points": n,
        "rmse": float(np.sqrt(sqerr / n)),
        "mae": float(abserr / n),
        "noisy_input_rmse": float(np.sqrt(baseline_sqerr / n)),
        "noisy_input_mae": float(baseline_abserr / n),
        "gaussian_nll": float(nll_total / n),
        "one_sigma_coverage": float(coverage / n),
        "device": device,
        "test_set_used": True,
        "by_held_out_regime": {},
    }
    for key, (se, ae, count) in by_scale.items():
        report["by_held_out_regime"][key] = {"rmse": float(np.sqrt(se / count)), "mae": float(ae / count), "valid_points": count}
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(report, indent=2), encoding="utf-8")

    # One representative high-noise held-out diagnostic; no plot is used for training.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sample = next(s for s in ds.samples if s["noise_scale"] >= 2.61)
    wl = np.asarray(sample["wavelength"], float)
    noisy = np.asarray(sample["noisy_flux"], float)
    clean = np.asarray(sample["clean_flux"], float)
    with torch.no_grad():
        x, _, _ = ds[ds.samples.index(sample)]
        mean, pred_log = model(x.unsqueeze(0).to(device))
        recovered = mean[0, 0].cpu().numpy()
        pred_sigma = torch.exp(pred_log[0, 0]).cpu().numpy()
    valid = np.isfinite(clean)
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(wl[valid], clean[valid], linewidth=1.4, label="Reference candidate")
    ax.plot(wl[valid], noisy[valid], linewidth=0.7, alpha=0.55, label="Held-out noisy")
    ax.plot(wl[valid], recovered[valid], linewidth=1.0, label="Neural recovery")
    ax.fill_between(wl[valid], recovered[valid]-pred_sigma[valid], recovered[valid]+pred_sigma[valid], alpha=0.15, label="Predicted ±1σ")
    ax.set_xlabel("Wavelength (µm)")
    ax.set_ylabel("Normalized flux")
    ax.set_title(f"OurionSpectra held-out recovery diagnostic | noise scale {sample['noise_scale']:.2f}×")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(Path(output_path).with_name("recovery_diagnostic.png"), dpi=180)
    plt.close(fig)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--test", default="data/wasp39b/training/test.json")
    p.add_argument("--output", default="artifacts/recovery_model/test_evaluation.json")
    args = p.parse_args()
    print(json.dumps(evaluate(args.model, args.test, args.output), indent=2))
