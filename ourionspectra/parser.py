"""
Reusable CSV parsing and metadata extraction routines for OurionSpectra.
Used by both the Tkinter Desktop UI and the FastAPI Backend.
"""

import csv
import re
from typing import Any, Dict, List, Optional, Tuple

from .config import WAVELENGTH_UNIT_TO_NM
from .science import guess_column, normalize_series


def is_float(value: Any) -> bool:
    """Check whether a value can be converted to float."""
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def extract_target_metadata(sample_text: str, headers: Optional[List[str]] = None, data_rows: Optional[List[List[str]]] = None) -> str:
    """Read a target identifier when explicitly specified in comments/metadata."""
    patterns = (
        r"^\s*[#;]\s*(?:target|object|object_name|target_name)\s*[:=]\s*(.+?)\s*$",
        r"^\s*(?:target|object|object_name|target_name)\s*[:=]\s*(.+?)\s*$",
    )
    for line in sample_text.splitlines()[:50]:
        for pattern in patterns:
            match = re.match(pattern, line, flags=re.IGNORECASE)
            if match:
                value = match.group(1).strip().strip('"').strip("'")
                if value:
                    return value

    if headers and data_rows:
        for idx, header in enumerate(headers):
            if header.strip().lower() in {"target", "target_name", "object", "object_name"}:
                values = {
                    row[idx].strip()
                    for row in data_rows
                    if len(row) > idx and row[idx].strip()
                }
                if len(values) == 1:
                    return next(iter(values))
    return "Not specified"


def sniff_and_read_csv(raw_text: str) -> Dict[str, Any]:
    """
    Sniff delimiter and parse header & raw data rows from CSV text.
    Returns:
        dict with keys: 'headers', 'data_rows', 'sample_text', 'target', 'has_header', 'delimiter'
    """
    if not raw_text or not raw_text.strip():
        raise ValueError("CSV content is empty.")

    lines = raw_text.splitlines()
    sample_text = "\n".join(lines[:50])

    try:
        dialect = csv.Sniffer().sniff(sample_text, delimiters=",;\t|")
        delimiter = dialect.delimiter
    except csv.Error:
        delimiter = ","

    try:
        has_header = csv.Sniffer().has_header(sample_text)
    except csv.Error:
        first_line = lines[0] if lines else ""
        first_cells = first_line.split(delimiter)
        has_header = any(not is_float(c.strip()) for c in first_cells)

    reader = csv.reader(raw_text.splitlines(), delimiter=delimiter)
    all_rows = [row for row in reader if any(cell.strip() for cell in row)]

    data_source_rows = []
    metadata_target_rows = []
    for row in all_rows:
        if not row or row[0].lstrip().startswith(("#", ";")):
            continue
        key = row[0].strip().lower()
        if key in {"target", "target_name", "object", "object_name"} and len(row) >= 2:
            value = row[1].strip()
            if value and not is_float(value):
                metadata_target_rows.append(value)
                continue
        data_source_rows.append(row)

    if not data_source_rows:
        raise ValueError("This CSV file has no spectrum data rows.")

    if has_header:
        headers = [h.strip() for h in data_source_rows[0]]
        data_rows = data_source_rows[1:]
    else:
        headers = [f"Column {i + 1}" for i in range(len(data_source_rows[0]))]
        data_rows = data_source_rows

    n_cols = len(headers)
    data_rows = [r for r in data_rows if len(r) == n_cols]
    if not data_rows:
        raise ValueError("Couldn't find rows matching the header's column count.")

    target = (
        metadata_target_rows[0]
        if metadata_target_rows
        else extract_target_metadata(sample_text, headers, data_rows)
    )

    return {
        "headers": headers,
        "data_rows": data_rows,
        "sample_text": sample_text,
        "target": target,
        "has_header": has_header,
        "delimiter": delimiter,
    }


def parse_spectrum_data(
    data_rows: List[List[str]],
    wl_idx: int = 0,
    flux_idx: int = 1,
    unit: str = "nm (nanometers)",
    normalize: bool = False,
) -> Tuple[List[float], List[float], int]:
    """
    Extract numeric wavelength and flux arrays from tabular string rows.
    Returns: (wavelengths, flux_values, skipped_count)
    """
    unit_factor = WAVELENGTH_UNIT_TO_NM.get(unit, 1.0)
    rows = []
    skipped = 0

    for r in data_rows:
        try:
            w = float(r[wl_idx].strip()) * unit_factor
            v = float(r[flux_idx].strip())
            rows.append((w, v))
        except (ValueError, IndexError):
            skipped += 1
            continue

    if len(rows) < 3:
        raise ValueError(
            "Couldn't find at least three valid numeric wavelength/flux rows in the selected columns."
        )

    # Some astronomy datasets store a batch of spectra by concatenating
    # complete spectra one after another.  Each spectrum then contains the
    # same wavelength grid, so a normal sort would interleave all spectra at
    # every wavelength and produce vertical spikes in the plot.  The desktop
    # workflow is a single-spectrum workflow, so when we can prove that the
    # file is made of repeated identical wavelength blocks, keep the first
    # complete spectrum and discard the remaining batch members.
    unique_wavelengths = []
    seen = set()
    for w, _ in rows:
        if w not in seen:
            seen.add(w)
            unique_wavelengths.append(w)

    block_size = len(unique_wavelengths)
    if block_size >= 3 and len(rows) % block_size == 0:
        repeats = len(rows) // block_size
        if repeats > 1:
            first_grid = [w for w, _ in rows[:block_size]]
            repeated_grid = all(
                [w for w, _ in rows[i * block_size:(i + 1) * block_size]] == first_grid
                for i in range(1, repeats)
            )
            if repeated_grid:
                rows = rows[:block_size]

    rows.sort(key=lambda r: r[0])
    wavelengths = [r[0] for r in rows]
    flux_values = [r[1] for r in rows]

    if normalize:
        flux_values = normalize_series(flux_values)

    return wavelengths, flux_values, skipped
