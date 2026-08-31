"""Opt-in experimental multi-target ML inference.

This module is intentionally not enabled by default. The Stage 15 scientific gate
found that the multi-target neural candidate does not consistently outperform the
transparent baseline, so production recovery remains unchanged.
"""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np, torch
from .fast_recovery import SpectralPointRecoveryMLP
from .science import moving_average

MODEL_PATH=Path(__file__).resolve().parents[1]/'artifacts/recovery_model/stage14_fast/model.pt'
ENABLED=os.getenv('OURIONSPECTRA_ENABLE_EXPERIMENTAL_ML','0').lower() in {'1','true','yes'}
_model=None

def _load():
 global _model
 if _model is None:
  if not MODEL_PATH.exists(): raise FileNotFoundError(MODEL_PATH)
  ck=torch.load(MODEL_PATH,map_location='cpu',weights_only=True);m=SpectralPointRecoveryMLP();m.load_state_dict(ck['model_state']);m.eval();_model=m
 return _model

def infer(wavelengths, noisy_flux, noise_level=0.3, restore_pct=100.0):
    wl = np.asarray(wavelengths, float)
    n = np.asarray(noisy_flux, float)
    base = np.asarray(moving_average(n.tolist(), window=7), float)
    sigma0 = max(float(noise_level) * 0.0077, 1e-4)
    resid = n - base
    robust = float(np.median(np.abs(resid - np.median(resid))) * 1.4826) if len(resid) > 2 else sigma0
    sigma = np.full_like(n, max(sigma0, robust), dtype=float)
    grad = np.gradient(n, wl) if len(wl) > 1 else np.zeros_like(n)
    curv = np.gradient(grad, wl) if len(wl) > 2 else np.zeros_like(n)
    X = np.column_stack([(wl - 0.34) / (5.17 - 0.34), n, sigma, base, n - base, grad * 0.01, curv * 0.0001, np.ones_like(n)]).astype('float32')
    m = _load()
    with torch.no_grad():
        pred, ls = m(torch.from_numpy(X))
        raw_pred = pred[:, 0].numpy()
        raw_ps = np.exp(ls[:, 0].numpy())

    alpha = max(0.0, min(1.0, float(restore_pct) / 100.0))
    final_pred = (1.0 - alpha) * n + alpha * raw_pred
    final_ps = raw_ps * alpha
    return final_pred, (final_pred - final_ps).tolist(), (final_pred + final_ps).tolist()

