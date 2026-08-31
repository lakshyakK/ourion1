"""Scientific benchmark suite for OurionSpectra recovery.

Compares the learned recovery model with transparent signal-processing baselines on the
held-out test set. The test set is never used for model fitting or checkpoint selection.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .recovery_model import SpectralRecoveryNet
from .train_recovery import SpectrumDataset


def _moving_average_masked(values: np.ndarray, valid: np.ndarray, radius: int) -> np.ndarray:
    out = np.full_like(values, np.nan, dtype=float)
    n = len(values)
    for i in range(n):
        if not valid[i]:
            continue
        lo, hi = max(0, i-radius), min(n, i+radius+1)
        m = valid[lo:hi]
        if m.any():
            out[i] = np.mean(values[lo:hi][m])
    return out


def _metrics(pred, target, valid):
    e = np.asarray(pred)[valid] - np.asarray(target)[valid]
    return {"rmse": float(np.sqrt(np.mean(e*e))), "mae": float(np.mean(np.abs(e))), "n": int(e.size)}


def run(model_path, test_path, output_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SpectralRecoveryNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True)["model_state"])
    model.eval()
    ds = SpectrumDataset(test_path)
    loader = DataLoader(ds, batch_size=32, shuffle=False)

    methods = {"raw_noisy": [], "moving_average_5": [], "moving_average_11": [], "neural_recovery": []}
    scale_bins = {"<0.45": [], "0.45-1.0": [], "1.0-2.0": [], ">=2.0": []}

    with torch.no_grad():
        cursor = 0
        for x, y, mask in loader:
            mean, _ = model(x.to(device))
            neural = mean[:, 0].cpu().numpy()
            x_np, y_np, m_np = x.numpy(), y.numpy()[:, 0], mask.numpy()[:, 0].astype(bool)
            for j in range(len(x_np)):
                noisy = x_np[j, 1]
                target = y_np[j]
                valid = m_np[j]
                wl = x_np[j, 0]
                del wl  # wavelength is already encoded by the model; baseline uses index-space smoothing.
                methods["raw_noisy"].append(_metrics(noisy, target, valid))
                methods["moving_average_5"].append(_metrics(_moving_average_masked(noisy, valid, 2), target, valid))
                methods["moving_average_11"].append(_metrics(_moving_average_masked(noisy, valid, 5), target, valid))
                methods["neural_recovery"].append(_metrics(neural[j], target, valid))
                scale = float(ds.samples[cursor + j]["noise_scale"])
                key = "<0.45" if scale < 0.45 else "0.45-1.0" if scale < 1.0 else "1.0-2.0" if scale < 2.0 else ">=2.0"
                scale_bins[key].append((noisy, neural[j], target, valid))
            cursor += len(x_np)

    def aggregate(items):
        n = sum(x["n"] for x in items)
        return {
            "rmse": float(np.sqrt(sum(x["rmse"]**2*x["n"] for x in items)/n)),
            "mae": float(sum(x["mae"]*x["n"] for x in items)/n),
            "valid_points": n,
        }

    report = {"test_samples": len(ds), "test_set_used_only_for_evaluation": True,
              "methods": {k: aggregate(v) for k,v in methods.items()}, "by_noise_scale": {}}
    for key, rows in scale_bins.items():
        if not rows:
            continue
        report["by_noise_scale"][key] = {}
        for name, idx in (("raw_noisy",0),("neural_recovery",1)):
            report["by_noise_scale"][key][name] = aggregate([_metrics(r[idx], r[2], r[3]) for r in rows])

    out = Path(output_path); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report

if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--test", default="data/wasp39b/training/test.json")
    p.add_argument("--output", default="artifacts/recovery_model/baseline_benchmark.json")
    args=p.parse_args()
    print(json.dumps(run(args.model,args.test,args.output),indent=2))
