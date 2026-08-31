"""
Pure functions for spectrum synthesis and simple statistics.
No Tkinter/Matplotlib imports here on purpose — keeps this module
trivially unit-testable.
"""

import math
import random

from .config import RANDOM_SEED

_RNG = random.Random(RANDOM_SEED)


def get_rng():
    """Expose the module-level RNG so the app can reuse the same stream."""
    return _RNG


def bump(x, center, amp, width):
    """Gaussian absorption bump used to build the synthetic 'true' spectrum."""
    return amp * math.exp(-((x - center) ** 2) / (2 * width * width))


def true_flux(wl_nm):
    """Synthetic ground-truth normalized flux at a given wavelength (nm)."""
    f = 1.0
    f -= bump(wl_nm, 1080, 0.03, 25)
    f -= bump(wl_nm, 1150, 0.05, 60)
    f -= bump(wl_nm, 1400, 0.11, 90)
    f -= bump(wl_nm, 1900, 0.07, 90)
    f -= bump(wl_nm, 2200, 0.02, 70)
    return f


def rmse(a, b):
    """Root-mean-square error between two equal-length sequences."""
    if not a or not b or len(a) != len(b):
        return 0.0
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)) / len(a))


def mae(a, b):
    """Mean absolute error between two equal-length sequences."""
    if not a or not b or len(a) != len(b):
        return 0.0
    return sum(abs(x - y) for x, y in zip(a, b)) / len(a)


def moving_average(arr, window=5):
    """Simple centered moving average, used as a 'ground truth' proxy for
    user-uploaded spectra where no true signal is known."""
    out = []
    half = window // 2
    for i in range(len(arr)):
        lo, hi = max(0, i - half), min(len(arr) - 1, i + half)
        out.append(sum(arr[lo:hi + 1]) / len(arr[lo:hi + 1]))
    return out


def normalize_series(values):
    """Scale values to roughly [-1, 1] / [0, 1] range by dividing by the
    largest absolute value. Used for raw physical flux units (W/m^2/um,
    Jy, etc.) that aren't already normalized."""
    if not values:
        return list(values)
    m = max(abs(v) for v in values)
    if m == 0:
        return list(values)
    return [v / m for v in values]


def generate_sample_spectrum(noise_level, wl_start=900.0, wl_end=2500.0, wl_step=22.0):
    """Build (wavelengths, true_spec, noisy_spec) for the built-in sample dataset."""
    wavelengths, true_spec, noisy_spec = [], [], []
    wl = wl_start
    while wl <= wl_end:
        w = round(wl, 1)
        wavelengths.append(w)
        t = true_flux(w)
        true_spec.append(t)
        noisy_spec.append(t + _RNG.gauss(0, 1) * noise_level * 0.05)
        wl += wl_step
    return wavelengths, true_spec, noisy_spec


def recover_spectrum(wavelengths, true_spec, noise_level):
    """Simulate an AI-recovered spectrum + uncertainty band around the truth."""
    recovered, unc_lower, unc_upper = [], [], []
    for i in range(len(wavelengths)):
        residual_noise = _RNG.gauss(0, 1) * max(noise_level, 0.05) * 0.008
        rec = true_spec[i] + residual_noise
        recovered.append(rec)
        sigma = 0.01 + noise_level * 0.02
        unc_lower.append(rec - sigma)
        unc_upper.append(rec + sigma)
    return recovered, unc_lower, unc_upper


def feature_confidence(strength, noise_level):
    """AI-confidence estimate (0-100) for a given feature strength/noise level."""
    conf = strength * 100 - noise_level * 35 + _RNG.uniform(-3, 3)
    return max(5, min(97, conf))


def status_from_confidence(conf):
    if conf >= 70:
        return "Detected"
    if conf >= 40:
        return "Tentative"
    return "Not Detected"


def guess_column(headers, keywords, exclude=None):
    """Best-effort guess of which column index matches a set of keywords
    (case-insensitive substring match). Falls back to the first available
    column that isn't `exclude`."""
    lowered = [h.lower() for h in headers]
    for i, h in enumerate(lowered):
        if i == exclude:
            continue
        if any(kw in h for kw in keywords):
            return i
    for i in range(len(headers)):
        if i != exclude:
            return i
    return 0
