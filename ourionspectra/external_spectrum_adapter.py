"""Stage 10: ingestion adapter for independently sourced published spectra.

This module converts the four-column format used by the Stellar Planet
transmission-spectrum archive into OurionSpectra's evaluation format without
modifying the source values. It is evaluation-only and never writes into the
training dataset.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict

import numpy as np


OURIONSPECTRA_COLUMNS = ("wavelength_micron", "normalized_flux", "normalized_uncertainty")


def _parse_rows(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        # Published archive files may be whitespace-delimited and may contain comments.
        rows = []
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").split()
            if len(parts) >= 4:
                try:
                    rows.append([float(x) for x in parts[:4]])
                except ValueError:
                    continue
        return np.asarray(rows, dtype=float)


def load_published_transmission(path: str | Path) -> Dict[str, np.ndarray]:
    """Load a published transmission spectrum in archive 4-column format.

    Expected columns:
      wavelength_um, wavelength_uncertainty_um, transit_depth, transit_depth_uncertainty

    Transit depth is used as the clean/reference flux-like quantity. No
    interpolation, smoothing, offset correction, or feature enhancement is applied.
    """
    path = Path(path)
    data = _parse_rows(path)
    if data.ndim != 2 or data.shape[0] < 20:
        raise ValueError("Published spectrum must contain at least 20 numeric rows")

    wavelength, wavelength_unc, depth, depth_unc = data.T
    finite = np.isfinite(wavelength) & np.isfinite(depth) & np.isfinite(depth_unc)
    finite &= depth_unc > 0
    if finite.sum() < 20:
        raise ValueError("Published spectrum needs at least 20 finite points with positive uncertainty")

    wavelength = wavelength[finite]
    wavelength_unc = wavelength_unc[finite]
    depth = depth[finite]
    depth_unc = depth_unc[finite]

    order = np.argsort(wavelength)
    wavelength = wavelength[order]
    wavelength_unc = wavelength_unc[order]
    depth = depth[order]
    depth_unc = depth_unc[order]

    if np.any(np.diff(wavelength) <= 0):
        raise ValueError("Published spectrum contains duplicate/non-increasing wavelengths")

    # Convert published transit depth D=(Rp/R*)^2 into the flux-like
    # representation used by OurionSpectra: F/F_star = 1-D. This is a
    # representation conversion, not an offset fit or normalization to WASP-39b.
    return {
        "wavelength_micron": wavelength,
        "normalized_flux": 1.0 - depth,
        "normalized_uncertainty": depth_unc,
        "wavelength_uncertainty_micron": wavelength_unc,
    }


def write_ourionspectra_csv(spectrum: Dict[str, np.ndarray], output: str | Path) -> Path:
    """Write an evaluation-only OurionSpectra-compatible CSV."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(OURIONSPECTRA_COLUMNS)
        for row in zip(spectrum["wavelength_micron"], spectrum["normalized_flux"], spectrum["normalized_uncertainty"]):
            writer.writerow([f"{float(row[0]):.10g}", f"{float(row[1]):.10g}", f"{float(row[2]):.10g}"])
    return output


def write_provenance(output: str | Path, source_name: str, source_url: str, publication_note: str) -> Path:
    """Record provenance for an externally supplied spectrum."""
    payload = {
        "stage": "10_external_spectrum_ingestion",
        "source_name": source_name,
        "source_url": source_url,
        "publication_note": publication_note,
        "role": "evaluation_only",
        "training_use": False,
        "validation_use": False,
        "model_selection_use": False,
        "ground_truth_language": "The published spectrum is a reference observation/candidate, not ground truth.",
        "feature_claims": "No atmospheric feature claims are produced by this adapter.",
        "transformations": ["column-format conversion", "transit-depth D=(Rp/R*)^2 to flux-like 1-D representation", "wavelength sorting only"],
        "limitations": [
            "Different published spectra may use different instruments, wavelength grids, system parameters, and reduction methods.",
            "Transit depth is not interchangeable with the WASP-39b normalized-flux scale; cross-target evaluation must therefore be scale-aware.",
            "A source must be independently downloaded and inspected before evaluation results are reported."
        ],
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output
