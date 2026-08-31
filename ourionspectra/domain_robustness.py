"""Stage 6: domain-robustness evaluation without fabricating new atmospheres.

This module deliberately does NOT synthesize alternative planetary spectra. It
creates observation-level domain shifts (calibration drift and correlation
length changes) around the same reference candidate and evaluates whether a
recovery model remains stable. The clean target remains the original reference
candidate throughout.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from .recovery_model import SpectralRecoveryNet
from .training_data import load_reference_spectrum, generate_noisy_realization
from .evaluate_baselines import _moving_average_masked, _metrics


def _smooth_field(n: int, step: float, length: float, rng: np.random.Generator) -> np.ndarray:
    white = rng.normal(size=n)
    radius = max(1, int(np.ceil(4 * length / step)))
    x = np.arange(-radius, radius + 1) * step
    kernel = np.exp(-0.5 * (x / length) ** 2)
    kernel /= kernel.sum()
    z = np.convolve(white, kernel, mode="same")
    z -= z.mean()
    return z / max(z.std(), 1e-12)


def make_domain_shifted_sample(ref, rng, noise_scale: float, drift_rms_fraction: float, correlation_length: float):
    """Generate noisy data, then apply an observation-only calibration drift.

    The drift is scaled by the local empirical uncertainty and is never applied
    to clean_flux. This represents an observational nuisance, not an invented
    atmospheric feature.
    """
    s = generate_noisy_realization(
        ref, rng, noise_scale=noise_scale, systematic_fraction=0.0,
        correlation_length_micron=correlation_length,
    )
    noisy = np.asarray(s["noisy_flux"], dtype=float)
    sigma = np.asarray(s["noise_sigma"], dtype=float)
    valid = np.isfinite(noisy) & np.isfinite(sigma)
    step = float(np.median(np.diff(ref.wavelengths)))
    field = _smooth_field(int(valid.sum()), step, correlation_length, rng)
    drift = np.zeros_like(noisy)
    drift[valid] = field * sigma[valid] * drift_rms_fraction
    noisy[valid] += drift[valid]
    s["noisy_flux"] = [None if not np.isfinite(x) else float(x) for x in noisy]
    s["domain_shift"] = {
        "type": "smooth_calibration_drift",
        "rms_in_empirical_sigma_units": float(drift_rms_fraction),
        "correlation_length_micron": float(correlation_length),
    }
    return s


def _model_prediction(model, sample):
    wl = np.asarray(sample["wavelength"], dtype=np.float32)
    noisy = np.asarray(sample["noisy_flux"], dtype=np.float32)
    sigma = np.asarray(sample["noise_sigma"], dtype=np.float32)
    clean = np.asarray(sample["clean_flux"], dtype=np.float32)
    valid = np.isfinite(clean) & np.isfinite(noisy) & np.isfinite(sigma)
    wl_norm = (wl - 0.63) / (5.17 - 0.63)
    baseline = _moving_average_masked(noisy, valid, radius=5)
    x = np.stack([np.nan_to_num(wl_norm), np.nan_to_num(noisy), np.nan_to_num(sigma), np.nan_to_num(baseline)], axis=0).astype(np.float32)
    with torch.no_grad():
        mean, log_sigma = model(torch.from_numpy(x[None, ...]))
    pred = mean[0, 0].numpy()
    pred[~valid] = np.nan
    return pred


def evaluate(model_path: str | Path, ref_path: str | Path = "data/wasp39b/wasp39b_reference.csv", seed: int = 2026):
    ref = load_reference_spectrum(ref_path)
    ckpt = torch.load(model_path, map_location="cpu")
    model = SpectralRecoveryNet(channels=32)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    rng = np.random.default_rng(seed)

    regimes = [
        {"name": "nominal_short", "noise_scale": 1.0, "drift": 0.15, "length": 0.04},
        {"name": "nominal_long", "noise_scale": 1.0, "drift": 0.15, "length": 0.20},
        {"name": "high_noise_short", "noise_scale": 2.8, "drift": 0.20, "length": 0.04},
        {"name": "high_noise_long", "noise_scale": 2.8, "drift": 0.20, "length": 0.20},
        {"name": "low_noise_unseen", "noise_scale": 0.30, "drift": 0.25, "length": 0.12},
    ]
    rows: List[Dict] = []
    for cfg in regimes:
        sample = make_domain_shifted_sample(ref, rng, cfg["noise_scale"], cfg["drift"], cfg["length"])
        clean = np.asarray(sample["clean_flux"], dtype=float)
        noisy = np.asarray(sample["noisy_flux"], dtype=float)
        pred = _model_prediction(model, sample)
        valid = np.isfinite(clean) & np.isfinite(noisy)
        smooth = _moving_average_masked(noisy, valid, radius=5)
        rows.append({
            **cfg,
            "neural": _metrics(pred, clean, valid),
            "moving_average_11": _metrics(smooth, clean, valid),
            "raw": _metrics(noisy, clean, valid),
        })
    return {"stage": "6_domain_robustness", "seed": seed, "target_is_original_reference_candidate": True, "regimes": rows,
            "scientific_note": "This benchmark tests observation-level domain shift only. It does not create alternative atmospheric spectra or claim planetary generalization."}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="artifacts/recovery_model/model.pt")
    p.add_argument("--output", default="artifacts/recovery_model/domain_robustness.json")
    args = p.parse_args()
    result = evaluate(args.model)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
