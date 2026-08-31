"""Scientific synthetic noisy-spectrum dataset generation for OurionSpectra.

The generator uses the WASP-39b reference-spectrum *candidate* and its empirical
per-channel uncertainty to create independent noisy observations. It preserves
all missing channels and never interpolates across gaps.
"""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

DEFAULT_REF_PATH = Path("data/wasp39b/wasp39b_reference.csv")
DEFAULT_OUTPUT_DIR = Path("data/wasp39b/training")


class ReferenceSpectrum:
    """Validated reference spectrum candidate."""

    def __init__(self, wavelengths, clean_flux, uncertainties, valid_mask, source_path):
        self.wavelengths = wavelengths
        self.clean_flux = clean_flux
        self.uncertainties = uncertainties
        self.valid_mask = valid_mask
        self.source_path = source_path
        self.n_total = len(wavelengths)
        self.n_valid = int(np.sum(valid_mask))
        self.n_gaps = self.n_total - self.n_valid


def load_reference_spectrum(csv_path: str | Path = DEFAULT_REF_PATH) -> ReferenceSpectrum:
    """Load the required reference CSV without filling or inventing missing data."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Reference spectrum file not found: {path}")

    wavelengths, flux, uncertainty, valid = [], [], [], []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"wavelength_micron", "normalized_flux", "normalized_uncertainty"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Reference CSV missing required columns: {sorted(missing)}")

        for row in reader:
            wl = float(row["wavelength_micron"])
            f = row["normalized_flux"].strip()
            u = row["normalized_uncertainty"].strip()
            wavelengths.append(wl)
            if f == "" or u == "":
                flux.append(np.nan)
                uncertainty.append(np.nan)
                valid.append(False)
            else:
                fv, uv = float(f), float(u)
                if not np.isfinite(fv) or not np.isfinite(uv) or uv <= 0:
                    flux.append(np.nan)
                    uncertainty.append(np.nan)
                    valid.append(False)
                else:
                    flux.append(fv)
                    uncertainty.append(uv)
                    valid.append(True)

    wavelengths = np.asarray(wavelengths, dtype=np.float64)
    clean_flux = np.asarray(flux, dtype=np.float64)
    uncertainties = np.asarray(uncertainty, dtype=np.float64)
    valid_mask = np.asarray(valid, dtype=bool)

    if len(wavelengths) < 2 or np.any(np.diff(wavelengths) <= 0):
        raise ValueError("Wavelength grid must be strictly increasing")

    return ReferenceSpectrum(wavelengths, clean_flux, uncertainties, valid_mask, str(path))


def _correlated_noise(
    wavelengths: np.ndarray,
    local_sigma: np.ndarray,
    rng: np.random.Generator,
    amplitude_fraction: float,
    correlation_length_micron: float,
) -> np.ndarray:
    """Generate a smooth zero-mean correlated component in uncertainty units.

    The process is constructed on the valid channels only. Its RMS is controlled
    relative to the empirical channel uncertainty, rather than an absolute flux
    scale, so heteroskedasticity is retained.
    """
    if amplitude_fraction <= 0:
        return np.zeros_like(local_sigma)

    # Random field followed by a Gaussian-kernel smoothing. The width is expressed
    # in wavelength units and therefore remains interpretable if the grid changes.
    white = rng.normal(size=len(wavelengths))
    step = float(np.median(np.diff(wavelengths)))
    radius = max(1, int(np.ceil(4.0 * correlation_length_micron / step)))
    x = np.arange(-radius, radius + 1, dtype=np.float64) * step
    kernel = np.exp(-0.5 * (x / correlation_length_micron) ** 2)
    kernel /= kernel.sum()
    smooth = np.convolve(white, kernel, mode="same")
    smooth -= smooth.mean()
    std = smooth.std()
    if std == 0:
        return np.zeros_like(local_sigma)
    smooth /= std

    # Local uncertainty scaling makes the systematic component wavelength-aware.
    return smooth * local_sigma * amplitude_fraction


def generate_noisy_realization(
    ref: ReferenceSpectrum,
    rng: np.random.Generator,
    noise_scale: float = 1.0,
    systematic_fraction: float = 0.0,
    correlation_length_micron: float = 0.08,
) -> Dict[str, Any]:
    """Generate one independent noisy observation from the reference candidate."""
    if noise_scale <= 0:
        raise ValueError("noise_scale must be > 0")
    if systematic_fraction < 0:
        raise ValueError("systematic_fraction must be >= 0")

    valid = ref.valid_mask
    base_sigma = ref.uncertainties[valid] * noise_scale
    white_noise = rng.normal(0.0, base_sigma)
    correlated = _correlated_noise(
        ref.wavelengths[valid],
        base_sigma,
        rng,
        systematic_fraction,
        correlation_length_micron,
    )
    residual = white_noise + correlated

    noisy = np.full(ref.n_total, np.nan, dtype=np.float64)
    sigma = np.full(ref.n_total, np.nan, dtype=np.float64)
    noisy[valid] = ref.clean_flux[valid] + residual
    # Marginal per-channel sigma includes both independent and correlated terms.
    sigma[valid] = base_sigma * np.sqrt(1.0 + systematic_fraction**2)

    estimated_snr = float(np.mean(np.abs(ref.clean_flux[valid])) / np.mean(sigma[valid]))
    return {
        "wavelength": [None if not np.isfinite(x) else round(float(x), 5) for x in ref.wavelengths],
        "clean_flux": [None if not np.isfinite(x) else float(x) for x in ref.clean_flux],
        "noisy_flux": [None if not np.isfinite(x) else float(x) for x in noisy],
        "noise_sigma": [None if not np.isfinite(x) else float(x) for x in sigma],
        "noise_scale": float(noise_scale),
        "estimated_snr": estimated_snr,
        "systematic_fraction": float(systematic_fraction),
        "correlation_length_micron": float(correlation_length_micron),
    }


def _sample_scale(rng: np.random.Generator, low: float, high: float) -> float:
    """Log-uniform sampling avoids over-populating the noisiest regime."""
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


def generate_dataset(
    ref_path: str | Path = DEFAULT_REF_PATH,
    n_train: int = 800,
    n_val: int = 150,
    n_test: int = 150,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], ReferenceSpectrum]:
    """Generate independent splits with deliberately different noise protocols.

    Train covers the central regime. Validation samples fixed benchmark regimes.
    Test contains held-out extreme regimes outside the train scale range, making
    the final evaluation a genuine robustness/generalization test.
    """
    ref = load_reference_spectrum(ref_path)
    master = np.random.default_rng(seed)
    train_rng = np.random.default_rng(int(master.integers(0, 2**63 - 1)))
    val_rng = np.random.default_rng(int(master.integers(0, 2**63 - 1)))
    test_rng = np.random.default_rng(int(master.integers(0, 2**63 - 1)))

    def make(split: str, count: int, rng: np.random.Generator):
        samples = []
        for idx in range(count):
            if split == "train":
                scale = _sample_scale(rng, 0.45, 2.60)
                systematic = float(rng.uniform(0.0, 0.30)) if rng.random() < 0.50 else 0.0
            elif split == "val":
                benchmarks = [0.60, 0.90, 1.20, 1.60, 2.00, 2.40]
                scale = float(benchmarks[idx % len(benchmarks)] + rng.uniform(-0.04, 0.04))
                systematic = 0.15 if idx % 2 == 0 else 0.0
            else:
                # Deliberately held-out low/high noise regimes.
                if idx % 2 == 0:
                    scale = float(rng.uniform(0.25, 0.44))
                else:
                    scale = float(rng.uniform(2.61, 3.40))
                systematic = float(rng.uniform(0.0, 0.35)) if idx % 2 == 0 else 0.20

            sample = generate_noisy_realization(ref, rng, scale, systematic)
            sample["sample_id"] = f"{split}_{idx:05d}"
            sample["split"] = split
            samples.append(sample)
        return samples

    return make("train", n_train, train_rng), make("val", n_val, val_rng), make("test", n_test, test_rng), ref


def _residual_correlation(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    ra = np.asarray(a["noisy_flux"], dtype=float) - np.asarray(a["clean_flux"], dtype=float)
    rb = np.asarray(b["noisy_flux"], dtype=float) - np.asarray(b["clean_flux"], dtype=float)
    mask = np.isfinite(ra) & np.isfinite(rb)
    if mask.sum() < 3:
        return 1.0
    return float(np.corrcoef(ra[mask], rb[mask])[0, 1])


def _split_statistics(samples: List[Dict[str, Any]]) -> Dict[str, float]:
    scales = np.asarray([s["noise_scale"] for s in samples])
    snr = np.asarray([s["estimated_snr"] for s in samples])
    return {
        "noise_scale_min": float(scales.min()),
        "noise_scale_max": float(scales.max()),
        "noise_scale_median": float(np.median(scales)),
        "estimated_snr_min": float(snr.min()),
        "estimated_snr_max": float(snr.max()),
        "estimated_snr_median": float(np.median(snr)),
    }


def save_dataset_and_metadata(
    train_samples: List[Dict[str, Any]],
    val_samples: List[Dict[str, Any]],
    test_samples: List[Dict[str, Any]],
    ref: ReferenceSpectrum,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    seed: int = 42,
) -> Dict[str, Any]:
    """Persist splits and complete provenance metadata."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, samples in (("train", train_samples), ("val", val_samples), ("test", test_samples)):
        with (out / f"{name}.json").open("w", encoding="utf-8") as handle:
            json.dump(samples, handle, allow_nan=False)

    all_samples = train_samples + val_samples + test_samples
    metadata = {
        "target": "WASP-39 b",
        "dataset_type": "Synthetic noisy-spectrum dataset for supervised recovery",
        "source_reference_file": ref.source_path,
        "number_of_samples": {
            "total": len(all_samples),
            "train": len(train_samples),
            "validation": len(val_samples),
            "test": len(test_samples),
        },
        "number_of_wavelength_points": ref.n_total,
        "valid_wavelength_points": ref.n_valid,
        "gap_wavelength_points": ref.n_gaps,
        "wavelength_range_micron": [float(ref.wavelengths.min()), float(ref.wavelengths.max())],
        "noise_generation": {
            "method": "Heteroskedastic Gaussian measurement noise scaled from normalized_uncertainty plus optional smooth correlated systematic component.",
            "base_uncertainty_median": float(np.nanmedian(ref.uncertainties)),
            "base_uncertainty_mean": float(np.nanmean(ref.uncertainties)),
            "correlated_component": "Gaussian-smoothed random field scaled locally by empirical uncertainty; zero mean per realization.",
            "correlation_length_micron": 0.08,
            "systematic_fraction_range": [0.0, 0.35],
        },
        "noise_snr_ranges": {
            "overall": _split_statistics(all_samples),
            "train": _split_statistics(train_samples),
            "validation": _split_statistics(val_samples),
            "test": _split_statistics(test_samples),
        },
        "train_validation_test_split": {"train": len(train_samples), "validation": len(val_samples), "test": len(test_samples)},
        "random_seed": seed,
        "leakage_prevention": {
            "independent_prng_streams": True,
            "test_noise_regimes_outside_train_range": True,
            "near_duplicate_check": "Residual correlation is checked by unit tests on independently generated samples.",
        },
        "preservation": {
            "reference_flux_unchanged": True,
            "missing_channels_interpolated": False,
            "nan_gap_channels_remain_null": True,
        },
        "scientific_limitations": [
            "The reference spectrum is a stitched observational reference candidate, not analytical ground truth.",
            "The generator models measurement noise statistically and cannot reproduce every JWST detector or reduction systematic.",
            "The correlated component is a controlled stochastic proxy, not a calibrated instrument noise covariance matrix.",
            "Synthetic training data are conditional on the supplied reference candidate and do not provide independent atmospheric truth.",
            "No atmospheric molecular features are added or claimed by the generator.",
        ],
    }
    with (out / "dataset_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, allow_nan=False)
    return metadata


def generate_diagnostic_plots(ref: ReferenceSpectrum, samples: List[Dict[str, Any]], output_path: str | Path):
    """Create clean-vs-noisy diagnostics at representative noise regimes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ordered = sorted(samples, key=lambda s: s["noise_scale"])
    indices = np.linspace(0, len(ordered) - 1, 4, dtype=int)
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    fig.suptitle("WASP-39 b Synthetic Noise Diagnostics — Reference Candidate vs Noisy Realizations")
    for i, idx in enumerate(indices):
        sample = ordered[idx]
        noisy = np.asarray(sample["noisy_flux"], dtype=float)
        sigma = np.asarray(sample["noise_sigma"], dtype=float)
        axes[i].plot(ref.wavelengths, ref.clean_flux, linewidth=1.3, label="Reference spectrum candidate")
        axes[i].plot(ref.wavelengths, noisy, linewidth=0.7, alpha=0.75, label="Synthetic noisy spectrum")
        axes[i].fill_between(ref.wavelengths, ref.clean_flux - sigma, ref.clean_flux + sigma, alpha=0.15, label="± marginal 1σ")
        axes[i].axvspan(3.715, 3.830, alpha=0.18, label="Preserved gap" if i == 0 else None)
        axes[i].set_ylabel("Normalized flux")
        axes[i].set_title(f"Scale {sample['noise_scale']:.2f}× | estimated SNR {sample['estimated_snr']:.1f}")
        axes[i].grid(alpha=0.25)
        axes[i].legend(fontsize=8)
    axes[-1].set_xlabel("Wavelength (µm)")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def build_training_dataset(
    ref_path: str | Path = DEFAULT_REF_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    n_train: int = 800,
    n_val: int = 150,
    n_test: int = 150,
    seed: int = 42,
) -> Dict[str, Any]:
    train, val, test, ref = generate_dataset(ref_path, n_train, n_val, n_test, seed)
    metadata = save_dataset_and_metadata(train, val, test, ref, output_dir, seed)
    generate_diagnostic_plots(ref, train + val + test, Path(output_dir) / "diagnostic_plots.png")
    return metadata


if __name__ == "__main__":
    build_training_dataset()
