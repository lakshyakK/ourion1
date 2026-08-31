"""Inference adapter for the OurionSpectra atmospheric-composition model.

The composition network takes 52 (wavelength, flux) samples flattened to
104 values (wavelength in microns, flux as the raw physical transit signal)
and predicts five log-abundances: H2O, CO2, CH4, CO, NH3.

Scaling statistics are baked into composition_scalers.npz (computed once at
training time) rather than re-derived from bundled training CSVs, so no
sklearn dependency or large training-data files are required at runtime.

This adapter deliberately reports model output as *predicted log abundances*,
not detection confidence. No atmospheric-feature confidence is fabricated.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
WEIGHTS_PATH = MODEL_DIR / "ourion_composition_model.pth"
SCALERS_PATH = MODEL_DIR / "composition_scalers.npz"

MOLECULES = ("H2O", "CO2", "CH4", "CO", "NH3")


class CompositionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(104, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 5),
        )

    def forward(self, x):
        return self.network(x)


_model = None
_x_mean = None
_x_std = None
_y_mean = None
_y_std = None
_template_wavelengths = None


def _load() -> None:
    global _model, _x_mean, _x_std, _y_mean, _y_std, _template_wavelengths
    if _model is not None:
        return
    for path in (WEIGHTS_PATH, SCALERS_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Missing composition model asset: {path}")

    scalers = np.load(SCALERS_PATH)
    _x_mean = scalers["x_mean"]
    _x_std = scalers["x_std"]
    _y_mean = scalers["y_mean"]
    _y_std = scalers["y_std"]
    _template_wavelengths = scalers["template_wavelengths"]

    _model = CompositionNet()
    state = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=True)
    _model.load_state_dict(state)
    _model.eval()


def is_available() -> bool:
    try:
        _load()
        return True
    except Exception:
        return False


def predict_composition(wavelengths: List[float], flux: List[float]) -> Dict[str, object]:
    """Predict the five composition parameters for a compatible spectrum.

    The model was trained on a fixed 52-point wavelength grid spanning
    roughly 0.55-6.48 microns. Inputs outside that coverage are rejected
    rather than silently extrapolated.
    """
    _load()
    wl = np.asarray(wavelengths, dtype=float)
    y = np.asarray(flux, dtype=float)
    if wl.size != y.size or wl.size < 2:
        raise ValueError("Wavelength and flux arrays must have the same length and contain at least two points.")
    if not np.all(np.isfinite(wl)) or not np.all(np.isfinite(y)):
        raise ValueError("Spectrum contains non-finite wavelength or flux values.")
    if np.any(y <= 0):
        raise ValueError("Composition model requires strictly positive flux values (it works in log-flux space).")

    order = np.argsort(wl)
    wl = wl[order]
    y = y[order]
    template = _template_wavelengths
    lo, hi = float(template.min()), float(template.max())
    tol = 1e-6 * max(abs(hi), 1.0)
    if float(wl.min()) > lo + tol or float(wl.max()) < hi - tol:
        raise ValueError(
            f"Composition model requires wavelength coverage from {lo:.3f} to {hi:.3f} microns; "
            f"received {wl.min():.3f} to {wl.max():.3f} microns."
        )

    model_flux = np.interp(template, wl, y)
    raw = np.concatenate([template, np.log10(model_flux)]).astype(np.float64)
    scaled = ((raw - _x_mean) / _x_std).reshape(1, 104).astype(np.float32)

    with torch.no_grad():
        scaled_pred = _model(torch.from_numpy(scaled)).numpy()[0]
    pred = scaled_pred * _y_std + _y_mean

    return {
        "available": True,
        "wavelength_min_um": lo,
        "wavelength_max_um": hi,
        "model": "OurionSpectra Atmospheric Composition Net",
        "parameters": [
            {"molecule": name, "log_abundance": float(value)}
            for name, value in zip(MOLECULES, pred)
        ],
    }

