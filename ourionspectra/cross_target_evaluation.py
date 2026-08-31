"""Stage 8: external-spectrum evaluation harness.

This module deliberately does not fabricate another planet's spectrum. It provides
an explicit, format-checked evaluation path for an independently supplied reference
candidate. The external target is NEVER used for training, calibration, or model
selection. This keeps cross-target validation scientifically honest.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from .recovery_model import SpectralRecoveryNet
from .evaluate_baselines import _moving_average_masked, _metrics

REQUIRED_COLUMNS = {"wavelength_micron", "normalized_flux", "normalized_uncertainty"}


def load_external_reference(path: str | Path) -> Dict[str, np.ndarray]:
    path = Path(path)
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("External reference file is empty")
    if not REQUIRED_COLUMNS.issubset(rows[0].keys()):
        raise ValueError(f"External reference must contain {sorted(REQUIRED_COLUMNS)}")
    wl = np.array([float(r["wavelength_micron"]) if r["wavelength_micron"] not in ("", "nan", "NaN") else np.nan for r in rows])
    flux = np.array([float(r["normalized_flux"]) if r["normalized_flux"] not in ("", "nan", "NaN") else np.nan for r in rows])
    unc = np.array([float(r["normalized_uncertainty"]) if r["normalized_uncertainty"] not in ("", "nan", "NaN") else np.nan for r in rows])
    if np.any(np.diff(wl[np.isfinite(wl)]) <= 0):
        raise ValueError("External wavelength grid must be strictly increasing")
    valid = np.isfinite(wl) & np.isfinite(flux) & np.isfinite(unc) & (unc > 0)
    if valid.sum() < 20:
        raise ValueError("External reference needs at least 20 valid spectral points")
    return {"wavelength": wl, "flux": flux, "uncertainty": unc, "valid": valid}


def _predict(model, wl, noisy, sigma, valid):
    # The model expects the same normalized wavelength coordinate used during training.
    lo, hi = float(np.nanmin(wl[valid])), float(np.nanmax(wl[valid]))
    denom = max(hi - lo, 1e-8)
    wnorm = (wl - lo) / denom
    baseline = _moving_average_masked(noisy, valid, radius=5)
    x = np.stack([np.nan_to_num(wnorm), np.nan_to_num(noisy), np.nan_to_num(sigma), np.nan_to_num(baseline)], axis=0).astype(np.float32)
    with torch.no_grad():
        mean, log_sigma = model(torch.from_numpy(x[None]))
    pred = mean[0, 0].numpy()
    pred_sigma = np.exp(log_sigma[0, 0].numpy())
    pred[~valid] = np.nan
    pred_sigma[~valid] = np.nan
    return pred, pred_sigma, baseline


def evaluate_file(model_path: str | Path, reference_path: str | Path, noise_scale: float = 1.0, seed: int = 2028) -> Dict:
    """Evaluate on a supplied independent reference candidate.

    This is evaluation-only. No target data are written into training datasets.
    The noise realization is generated from that file's own empirical uncertainty.
    """
    ref = load_external_reference(reference_path)
    rng = np.random.default_rng(seed)
    valid = ref["valid"]
    sigma = ref["uncertainty"].copy()
    noisy = ref["flux"].copy()
    noisy[valid] += rng.normal(0.0, noise_scale * sigma[valid])

    model = SpectralRecoveryNet(channels=32)
    ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    pred, pred_sigma, baseline = _predict(model, ref["wavelength"], noisy, noise_scale * sigma, valid)

    return {
        "stage": "8_external_reference_evaluation",
        "reference_file": str(reference_path),
        "training_used": False,
        "validation_used": False,
        "model_selection_used": False,
        "noise_scale": float(noise_scale),
        "valid_points": int(valid.sum()),
        "raw": _metrics(noisy, ref["flux"], valid),
        "moving_average_11": _metrics(baseline, ref["flux"], valid),
        "neural": _metrics(pred, ref["flux"], valid),
        "uncertainty_1sigma_coverage": float(np.mean(np.abs(pred[valid] - ref["flux"][valid]) <= pred_sigma[valid])),
        "scientific_note": "An independently supplied reference candidate is evaluated only. This result must not be described as ground truth or as evidence of atmospheric-feature recovery without separate scientific validation.",
    }


def write_protocol(path: str | Path = "artifacts/recovery_model/stage8_external_evaluation_protocol.json"):
    protocol = {
        "stage": "8_external_reference_evaluation",
        "purpose": "Provide a reproducible pathway for independent-spectrum validation without fabricating a second planet spectrum.",
        "required_columns": sorted(REQUIRED_COLUMNS),
        "data_split_rule": "External reference files are evaluation-only and must never be copied into train/val datasets.",
        "ground_truth_language": "The supplied spectrum remains a reference candidate, not ground truth.",
        "atmospheric_feature_rule": "No molecular features are added, amplified, or claimed by this module.",
        "current_status": "Awaiting an independently sourced reference candidate with provenance and uncertainty columns.",
    }
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    return protocol

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="artifacts/recovery_model/model.pt")
    p.add_argument("--reference")
    p.add_argument("--output", default="artifacts/recovery_model/stage8_external_evaluation.json")
    a = p.parse_args()
    if a.reference:
        result = evaluate_file(a.model, a.reference)
        Path(a.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
    else:
        print(json.dumps(write_protocol(), indent=2))
