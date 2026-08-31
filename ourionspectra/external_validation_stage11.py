"""Stage 11: independent-target external validation.

Uses an independently published HAT-P-1b HST/WFC3 transmission spectrum as an
evaluation-only reference candidate. The published Rp/R* values are converted to
transit depth and then to OurionSpectra's flux-like representation. Synthetic noise is
added ONLY for a controlled stress test using the published measurement uncertainty.
The target is never used for training, validation, calibration, or model selection.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from .evaluate_baselines import _moving_average_masked, _metrics
from .recovery_model import SpectralRecoveryNet


def load_external(path: str | Path):
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    required = {"wavelength_micron", "rp_over_rstar", "rp_over_rstar_uncertainty"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Missing required columns: {sorted(required)}")
    wl = np.array([float(r["wavelength_micron"]) for r in rows], dtype=float)
    r = np.array([float(r["rp_over_rstar"]) for r in rows], dtype=float)
    sr = np.array([float(r["rp_over_rstar_uncertainty"]) for r in rows], dtype=float)
    if np.any(~np.isfinite(wl)) or np.any(~np.isfinite(r)) or np.any(~np.isfinite(sr)) or np.any(sr <= 0):
        raise ValueError("External spectrum contains invalid values")
    if np.any(np.diff(wl) <= 0):
        raise ValueError("Wavelengths must be strictly increasing")
    depth = r * r
    depth_sigma = 2.0 * r * sr
    flux = 1.0 - depth
    return wl, flux, depth_sigma


def predict(model, wl, noisy, sigma):
    valid = np.isfinite(wl) & np.isfinite(noisy) & np.isfinite(sigma) & (sigma > 0)
    # Match the coordinate convention used during WASP-39b training.
    wnorm = (wl - 0.63) / (5.17 - 0.63)
    baseline = _moving_average_masked(noisy, valid, radius=5)
    x = np.stack([wnorm, noisy, sigma, baseline], axis=0).astype(np.float32)
    with torch.no_grad():
        mean, log_sigma = model(torch.from_numpy(x[None]))
    pred = mean[0, 0].numpy()
    pred_sigma = np.exp(log_sigma[0, 0].numpy())
    return pred, pred_sigma, baseline


def evaluate(model_path, reference_path, output_dir, seed=2028, realizations_per_scale=100):
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    wl, clean, sigma = load_external(reference_path)
    model = SpectralRecoveryNet(channels=32)
    ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    rng = np.random.default_rng(seed)
    scales = [0.5, 1.0, 1.5, 2.0]
    aggregate = {}
    coverage = {}
    examples = []

    for scale in scales:
        raw_m, ma_m, neural_m = [], [], []
        cov = []
        for i in range(realizations_per_scale):
            noisy = clean + rng.normal(0.0, scale * sigma)
            pred, pred_sigma, baseline = predict(model, wl, noisy, scale * sigma)
            valid = np.isfinite(clean)
            raw_m.append(_metrics(noisy, clean, valid))
            ma_m.append(_metrics(baseline, clean, valid))
            neural_m.append(_metrics(pred, clean, valid))
            cov.append(float(np.mean(np.abs(pred[valid] - clean[valid]) <= pred_sigma[valid])))
            if i == 0:
                examples.append((scale, noisy, pred, baseline, pred_sigma))
        def mean_metric(items, key):
            return float(np.mean([x[key] for x in items]))
        aggregate[str(scale)] = {
            "raw": {"rmse": mean_metric(raw_m, "rmse"), "mae": mean_metric(raw_m, "mae")},
            "moving_average_11": {"rmse": mean_metric(ma_m, "rmse"), "mae": mean_metric(ma_m, "mae")},
            "neural": {"rmse": mean_metric(neural_m, "rmse"), "mae": mean_metric(neural_m, "mae")},
            "neural_vs_ma11_rmse_improvement": float((mean_metric(ma_m, "rmse") - mean_metric(neural_m, "rmse")) / mean_metric(ma_m, "rmse")),
            "realizations": realizations_per_scale,
        }
        coverage[str(scale)] = float(np.mean(cov))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=False)
    for ax, (scale, noisy, pred, baseline, pred_sigma) in zip(axes.ravel(), examples):
        ax.plot(wl, clean, label="Published reference candidate")
        ax.plot(wl, noisy, label=f"Synthetic noisy ({scale}× σ)", alpha=0.55)
        ax.plot(wl, baseline, label="Moving-average baseline")
        ax.plot(wl, pred, label="OurionSpectra recovery")
        ax.fill_between(wl, pred - pred_sigma, pred + pred_sigma, alpha=0.15, label="Predicted ±1σ")
        ax.set_title(f"HAT-P-1b external stress test — {scale}× uncertainty")
        ax.set_ylabel("Flux-like representation")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7)
    axes[-1, 0].set_xlabel("Wavelength (µm)")
    axes[-1, 1].set_xlabel("Wavelength (µm)")
    fig.tight_layout()
    plot_path = out / "stage11_external_validation.png"
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)

    report = {
        "stage": "11_independent_target_external_validation",
        "target": "HAT-P-1b",
        "source_reference": "Wakeford et al. 2013, MNRAS 435, 3481; HST/WFC3 G141",
        "source_url": "https://academic.oup.com/mnras/article/435/4/3481/1034590",
        "source_type": "peer-reviewed published transmission spectrum",
        "valid_points": int(len(wl)),
        "wavelength_range_micron": [float(wl.min()), float(wl.max())],
        "noise_method": "independent Gaussian perturbations using propagated published Rp/R* uncertainty; no target-specific calibration",
        "noise_scales": scales,
        "realizations_per_scale": realizations_per_scale,
        "training_used": False,
        "validation_used": False,
        "model_selection_used": False,
        "ground_truth": False,
        "feature_claims": False,
        "results": aggregate,
        "uncertainty_1sigma_coverage": coverage,
        "plot": str(plot_path),
        "limitations": [
            "This is an external-target synthetic-noise stress test, not recovery of the original raw HAT-P-1b time-series data.",
            "The published spectrum has a much coarser grid and narrower wavelength range than the WASP-39b training grid.",
            "Cross-target absolute transit-depth levels can differ because of system parameters and reduction choices; no offset fitting was applied.",
            "The result does not establish generalization to all exoplanets or prove atmospheric-feature recovery.",
        ],
    }
    report_path = out / "stage11_external_validation.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
