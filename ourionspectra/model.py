"""
This is the ONE file to replace when you plug in a real ML model.

`run_recovery_model` is the only function the rest of the app calls. Its
signature is fixed — the UI, history, and PDF export all depend on this
exact input/output shape — but its *implementation* is yours to replace
entirely.

    Input:
        wavelengths: list[float]   — wavelength in nm, sorted ascending
        noisy_flux:  list[float]   — observed/noisy normalized flux, same length
        noise_level: float         — 0.0 (clean) to 1.0 (very noisy), a UI hint
                                      you can use or ignore

    Output:  (recovered_flux, uncertainty_lower, uncertainty_upper)
        Three lists, each the same length as `wavelengths`.
        - recovered_flux: your model's best-guess denoised/recovered spectrum
        - uncertainty_lower / uncertainty_upper: a ±band around recovered_flux
          for the shaded uncertainty region in the chart. If your model
          doesn't produce uncertainty, just return recovered_flux ± a small
          constant, or recovered_flux twice (zero-width band).

Nothing else in the app needs to change. Swap the body of this function
for a real model's inference call (load weights once at import time,
run forward pass here, etc.) and everything downstream — the chart,
RMSE stats, feature confidence, history, and PDF export — keeps working.

--------------------------------------------------------------------
DEFAULT IMPLEMENTATION (placeholder, not a real ML model)
--------------------------------------------------------------------
Right now this just runs a smoothing denoiser (a wider moving average)
over the noisy input and derives a simple uncertainty band from the
residual spread. It's good enough to demo the UI end-to-end, but it is
NOT a trained model and shouldn't be treated as scientifically meaningful.
"""

import math
import statistics
import os

from .science import moving_average
from .experimental_inference import ENABLED as EXPERIMENTAL_ML_ENABLED, infer as experimental_infer

# Shown in the "Recovery Model" field of the Dataset & Model Info card.
# Update this string when the real trained model is wired in.
MODEL_NAME = "OurionSpectra Validated Spectral Recovery + Trained Composition Net"
if EXPERIMENTAL_ML_ENABLED:
    MODEL_NAME = "OurionSpectra Multi-Target ML (experimental)"


def run_recovery_model(wavelengths, noisy_flux, noise_level=0.3, restore_pct=100.0):
    """Recover a spectrum through the stable app interface with configurable restoration percentage.

    Production/default mode intentionally remains the transparent baseline.
    The multi-target neural candidate is opt-in because Stage 15 did not approve
    it for default scientific use.
    """
    if EXPERIMENTAL_ML_ENABLED:
        return experimental_infer(wavelengths, noisy_flux, noise_level, restore_pct=restore_pct)

    window = 7
    raw_recovered = moving_average(noisy_flux, window=window)
    residuals = [n - r for n, r in zip(noisy_flux, raw_recovered)]
    sigma = statistics.pstdev(residuals) if len(residuals) > 1 else 0.01
    sigma = max(sigma, 0.004)

    alpha = max(0.0, min(1.0, float(restore_pct) / 100.0))
    recovered = [
        (1.0 - alpha) * n + alpha * r
        for n, r in zip(noisy_flux, raw_recovered)
    ]
    effective_sigma = sigma * alpha
    uncertainty_lower = [r - effective_sigma for r in recovered]
    uncertainty_upper = [r + effective_sigma for r in recovered]
    return recovered, uncertainty_lower, uncertainty_upper



# Major exoplanet spectral absorption bands (wavelength in nm, bandwidth in nm)
ATMOSPHERIC_FEATURE_TEMPLATES = [
    {"name": "H2O (Water Vapor)", "wl_nm": 1150.0, "half_width": 45.0, "weight": 1.0},
    {"name": "H2O (Water Vapor)", "wl_nm": 1400.0, "half_width": 70.0, "weight": 1.4},
    {"name": "H2O (Water Vapor)", "wl_nm": 1900.0, "half_width": 80.0, "weight": 1.2},
    {"name": "CO2 (Carbon Dioxide)", "wl_nm": 2000.0, "half_width": 50.0, "weight": 1.1},
    {"name": "CO2 (Carbon Dioxide)", "wl_nm": 2700.0, "half_width": 80.0, "weight": 1.3},
    {"name": "CO2 (Carbon Dioxide)", "wl_nm": 4300.0, "half_width": 120.0, "weight": 1.5},
    {"name": "CH4 (Methane)", "wl_nm": 1660.0, "half_width": 45.0, "weight": 1.0},
    {"name": "CH4 (Methane)", "wl_nm": 2300.0, "half_width": 60.0, "weight": 1.2},
    {"name": "CH4 (Methane)", "wl_nm": 3300.0, "half_width": 90.0, "weight": 1.3},
    {"name": "CO (Carbon Monoxide)", "wl_nm": 2350.0, "half_width": 55.0, "weight": 1.0},
    {"name": "CO (Carbon Monoxide)", "wl_nm": 4650.0, "half_width": 100.0, "weight": 1.3},
    {"name": "NH3 (Ammonia)", "wl_nm": 1500.0, "half_width": 40.0, "weight": 1.0},
    {"name": "NH3 (Ammonia)", "wl_nm": 2000.0, "half_width": 60.0, "weight": 1.1},
    {"name": "Na (Sodium Doublet)", "wl_nm": 589.0, "half_width": 15.0, "weight": 1.5},
    {"name": "K (Potassium)", "wl_nm": 769.0, "half_width": 20.0, "weight": 1.3},
]


def detect_atmospheric_features(wavelengths, recovered_flux, noise_level=0.3):
    """
    Perform spectroscopic feature detection on recovered transmission spectrum.
    Measures band absorption depth relative to local continuum to estimate detection
    significance (SNR) and confidence.
    """
    if not wavelengths or not recovered_flux or len(wavelengths) != len(recovered_flux):
        return []

    wl_min = min(wavelengths)
    wl_max = max(wavelengths)
    if wl_max - wl_min < 20.0:
        return []

    # Estimate overall local flux dispersion / noise floor
    flux_diffs = [abs(recovered_flux[i] - recovered_flux[i - 1]) for i in range(1, len(recovered_flux))]
    noise_floor = (statistics.median(flux_diffs) if flux_diffs else 0.01) * 0.707 + (float(noise_level) * 0.005)
    noise_floor = max(noise_floor, 1e-4)

    results = []
    for tpl in ATMOSPHERIC_FEATURE_TEMPLATES:
        center = tpl["wl_nm"]
        hw = tpl["half_width"]

        # Only evaluate bands fully within observation wavelength coverage
        if center - hw < wl_min or center + hw > wl_max:
            continue

        # In-band indices
        in_band = [i for i, w in enumerate(wavelengths) if abs(w - center) <= hw]
        # Out-of-band local continuum indices (flanking sidebands)
        left_band = [i for i, w in enumerate(wavelengths) if (center - 2.5 * hw) <= w < (center - hw)]
        right_band = [i for i, w in enumerate(wavelengths) if (center + hw) < w <= (center + 2.5 * hw)]

        if not in_band or (not left_band and not right_band):
            continue

        # Continuum level from surrounding points
        continuum_pts = left_band + right_band
        continuum = sum(recovered_flux[i] for i in continuum_pts) / len(continuum_pts)
        band_min_flux = min(recovered_flux[i] for i in in_band)

        # Absorption depth (in transmission spectroscopy, absorption = dip below continuum)
        depth = continuum - band_min_flux
        if depth <= 0:
            confidence = 10.0
            status = "Not Detected"
        else:
            snr = (depth / noise_floor) * tpl["weight"]
            # Convert SNR to a 0-99 confidence percentage
            confidence = min(98.0, max(5.0, 100.0 / (1.0 + math.exp(-0.9 * (snr - 2.2)))))
            if snr >= 3.0 and confidence >= 70.0:
                status = "Detected"
            elif snr >= 1.8 and confidence >= 45.0:
                status = "Tentative"
            else:
                status = "Not Detected"

        results.append({
            "name": tpl["name"],
            "wl_nm": center,
            "confidence": round(confidence, 1),
            "status": status,
        })

    # Sort: Detected first, then Tentative, then by wavelength
    status_order = {"Detected": 0, "Tentative": 1, "Not Detected": 2}
    results.sort(key=lambda item: (status_order.get(item["status"], 3), item["wl_nm"]))
    return results

