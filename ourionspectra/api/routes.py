"""
FastAPI route definitions for OurionSpectra backend.
"""

import csv
import io
import os
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles


from ourionspectra.config import NAVY, BLUE_ACCENT, GRAY_LINE, TEXT_SUB, WAVELENGTH_UNIT_TO_NM
from ourionspectra.science import (
    generate_sample_spectrum,
    rmse,
    mae,
    guess_column,
    normalize_series,
)
from ourionspectra.model import run_recovery_model, detect_atmospheric_features, MODEL_NAME
from ourionspectra.composition_model import predict_composition
from ourionspectra.parser import sniff_and_read_csv, parse_spectrum_data, extract_target_metadata
from ourionspectra import storage
from ourionspectra.report import export_report_pdf

from .schemas import (
    HealthResponse,
    SpectrumSampleRequest,
    SpectrumSampleResponse,
    CSVParsePreviewResponse,
    CSVProcessRequest,
    SpectrumDataResponse,
    RecoveryRequest,
    RecoveryResponse,
    RecoveryMetrics,
    FeaturesRequest,
    FeaturesResponse,
    FeatureItem,
    CompositionResult,
    SessionSummary,
    SessionDetail,
    ExportCSVRequest,
    ExportPDFRequest,
)

app = FastAPI(
    title="OurionSpectra API",
    description="Backend service for Exoplanet Atmospheric Spectrum Recovery (SICT037)",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).resolve().parent / "static"
INDEX_HTML = STATIC_DIR / "index.html"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")



@app.get("/", include_in_schema=False)
@app.get("/app", include_in_schema=False)
def serve_web_app():
    """Serve the interactive OurionSpectra Web Application."""
    if INDEX_HTML.exists():
        return FileResponse(str(INDEX_HTML))
    return {"message": "OurionSpectra API running. Open /docs for Swagger documentation."}


@app.get("/api/health", response_model=HealthResponse, tags=["General"])

def get_health():
    """Health check and model capability details."""
    return HealthResponse(
        status="ok",
        app_name="OURIONSPECTRA API",
        version="1.0.0",
        model_name=MODEL_NAME,
    )


@app.post("/api/sample", response_model=SpectrumSampleResponse, tags=["Spectrum"])
def get_sample_spectrum(payload: Optional[SpectrumSampleRequest] = None):
    """Generate a synthetic near-IR transit spectrum with realistic noise."""
    if payload is None:
        payload = SpectrumSampleRequest()

    try:
        wavelengths, reference, noisy = generate_sample_spectrum(
            noise_level=payload.noise_level,
            wl_start=payload.wl_start,
            wl_end=payload.wl_end,
            wl_step=payload.wl_step,
        )
        return SpectrumSampleResponse(
            wavelengths=wavelengths,
            reference_spec=reference,
            noisy_spec=noisy,
            noise_level=payload.noise_level,
            dataset="Sample Spectrum (simulated near-IR transit)",
            target="Not specified",
            instrument="Simulated NIR spectrograph",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate sample spectrum: {str(e)}")


@app.post("/api/spectrum/preview-file", response_model=CSVParsePreviewResponse, tags=["Spectrum"])
async def preview_csv_file(file: UploadFile = File(...)):
    """Upload a CSV spectrum file to detect headers, delimiter, target, and suggested columns."""
    try:
        content_bytes = await file.read()
        text = content_bytes.decode("utf-8-sig")
    except UnicodeDecodeError:
        try:
            text = content_bytes.decode("latin-1")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to decode file content: {str(e)}")

    try:
        parsed = sniff_and_read_csv(text)
        headers = parsed["headers"]
        data_rows = parsed["data_rows"]
        target = parsed["target"]

        wl_idx = guess_column(headers, ("wavelength", "wl", "lambda", "wave"))
        flux_idx = guess_column(headers, ("flux", "flam", "fnu", "flux_norm", "value"), exclude=wl_idx)

        unit_hint = " ".join(headers).lower()
        default_unit = (
            "µm (microns)"
            if ("micron" in unit_hint or "_micron" in unit_hint or "um" in unit_hint)
            else "nm (nanometers)"
        )

        return CSVParsePreviewResponse(
            headers=headers,
            preview_rows=data_rows[:5],
            total_data_rows=len(data_rows),
            target=target,
            suggested_wl_idx=wl_idx,
            suggested_flux_idx=flux_idx,
            suggested_unit=default_unit,
            available_units=list(WAVELENGTH_UNIT_TO_NM.keys()),
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


def _is_float_cell(value):
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False



@app.post("/api/spectrum/process-csv", response_model=SpectrumDataResponse, tags=["Spectrum"])
def process_csv_content(payload: CSVProcessRequest):
    """Process raw CSV content with specific column indices, units, and normalization."""
    try:
        parsed = sniff_and_read_csv(payload.csv_content)
        data_rows = parsed["data_rows"]
        target = payload.target if payload.target else parsed["target"]

        wavelengths, flux_values, skipped = parse_spectrum_data(
            data_rows=data_rows,
            wl_idx=payload.wl_idx,
            flux_idx=payload.flux_idx,
            unit=payload.unit,
            normalize=payload.normalize,
        )

        warnings = []
        if skipped > 0:
            warnings.append(f"{skipped} row(s) were skipped (missing or non-numeric values).")
        numeric_row_count = sum(
            1 for row in data_rows
            if len(row) > max(payload.wl_idx, payload.flux_idx)
            and _is_float_cell(row[payload.wl_idx]) and _is_float_cell(row[payload.flux_idx])
        )
        if numeric_row_count > len(wavelengths) and len(wavelengths) >= 3:
            warnings.append(
                f"Detected a repeated wavelength grid. Loaded the first complete spectrum "
                f"({len(wavelengths)} points) for single-spectrum analysis."
            )
        wl_lo, wl_hi = wavelengths[0], wavelengths[-1]
        if wl_lo < 300 or wl_hi > 30000:
            warnings.append(
                f"Wavelength range ({wl_lo:.0f}–{wl_hi:.0f} nm) looks unusual — check the unit selected."
            )

        return SpectrumDataResponse(
            wavelengths=wavelengths,
            noisy_spec=flux_values,
            target=target,
            dataset=payload.dataset_name or "Uploaded Spectrum",
            instrument=payload.instrument or "User-supplied CSV",
            skipped_rows=skipped,
            warnings=warnings,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/spectrum/recover", response_model=RecoveryResponse, tags=["Recovery"])
def recover_spectrum(payload: RecoveryRequest):
    """
    Run spectrum recovery through the ML model interface and compute metrics
    when ground truth is available.
    """
    if len(payload.wavelengths) != len(payload.noisy_spec):
        raise HTTPException(status_code=400, detail="Wavelengths and noisy_spec lengths must match.")

    try:
        raw_recovered, raw_unc_lower, raw_unc_upper = run_recovery_model(
            wavelengths=payload.wavelengths,
            noisy_flux=payload.noisy_spec,
            noise_level=payload.noise_level,
        )

        alpha = max(0.0, min(1.0, payload.restore_percent / 100.0))
        recovered = [
            (1.0 - alpha) * n + alpha * r
            for n, r in zip(payload.noisy_spec, raw_recovered)
        ]

        if raw_unc_lower and raw_unc_upper:
            unc_width = [
                (u - l) * 0.5 * alpha
                for l, u in zip(raw_unc_lower, raw_unc_upper)
            ]
            unc_lower = [r - w for r, w in zip(recovered, unc_width)]
            unc_upper = [r + w for r, w in zip(recovered, unc_width)]
        else:
            unc_lower = []
            unc_upper = []

        # Compute metrics if reference spectrum is present and valid
        metrics_dict: dict = {}
        has_ref = (
            payload.reference_spec is not None
            and len(payload.reference_spec) == len(payload.wavelengths)
        )

        if has_ref:
            before = rmse(payload.noisy_spec, payload.reference_spec)
            after = rmse(recovered, payload.reference_spec)
            after_mae = mae(recovered, payload.reference_spec)
            metrics_dict["rmse_before"] = before
            metrics_dict["rmse_after"] = after
            metrics_dict["mae"] = after_mae
            if before:
                metrics_dict["recovery_improvement"] = (1 - after / before) * 100.0

        features_data = detect_atmospheric_features(
            wavelengths=payload.wavelengths,
            recovered_flux=recovered,
            noise_level=payload.noise_level,
        )


        composition = None
        try:
            # API wavelengths are normalized to nm by the CSV parser. The
            # uploaded composition model was trained on microns.
            wl_um = [float(w) / 1000.0 for w in payload.wavelengths]
            composition = predict_composition(wl_um, recovered)
        except ValueError:
            # Coverage mismatch is a normal condition for spectra that do not
            # span the model's fixed 0.55–7.28 µm training grid. Keep recovery
            # successful and report composition as unavailable.
            composition = {"available": False, "parameters": []}
        except Exception:
            composition = {"available": False, "parameters": []}

        session_id = None
        if payload.save_session:
            session_data = {
                "wavelengths": payload.wavelengths,
                "noisy_spec": payload.noisy_spec,
                "reference_spec": payload.reference_spec if has_ref else [],
                "recovered_spec": recovered,
                "unc_lower": unc_lower,
                "unc_upper": unc_upper,
                "dataset": payload.dataset or "Unnamed Dataset",
                "target": payload.target or "Not specified",
                "instrument": payload.instrument or "User-supplied CSV",
                "noise_level": payload.noise_level,
                "has_reference": has_ref,
                "features": features_data,
                "metrics": metrics_dict,
                "metrics_reference_available": has_ref,
            }
            session_id = storage.save_session(session_data)

        return RecoveryResponse(
            session_id=session_id,
            wavelengths=payload.wavelengths,
            noisy_spec=payload.noisy_spec,
            reference_spec=payload.reference_spec,
            recovered_spec=recovered,
            unc_lower=unc_lower,
            unc_upper=unc_upper,
            metrics=RecoveryMetrics(**metrics_dict),
            features=[FeatureItem(**f) for f in features_data],
            model_name=MODEL_NAME,
            composition=CompositionResult(**composition) if composition else None,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recovery failed: {str(e)}")


@app.post("/api/spectrum/features", response_model=FeaturesResponse, tags=["Features"])
def get_atmospheric_features(payload: FeaturesRequest):
    """Detect atmospheric molecules from recovered spectrum."""
    if len(payload.wavelengths) != len(payload.recovered_spec):
        raise HTTPException(status_code=400, detail="Wavelengths and recovered_spec lengths must match.")

    try:
        features = detect_atmospheric_features(
            wavelengths=payload.wavelengths,
            recovered_flux=payload.recovered_spec,
            noise_level=payload.noise_level,
        )
        return FeaturesResponse(features=[FeatureItem(**f) for f in features])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/sessions", response_model=List[SessionSummary], tags=["Sessions"])
def get_all_sessions():
    """Retrieve metadata list of all saved sessions, newest first."""
    try:
        sessions = storage.list_sessions()
        return [SessionSummary(**s) for s in sessions]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list sessions: {str(e)}")


@app.get("/api/sessions/{session_id}", response_model=SessionDetail, tags=["Sessions"])
def get_session(session_id: str):
    """Retrieve full data for a specific saved session by ID."""
    try:
        session = storage.load_session(session_id)
        return SessionDetail(**session)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Session not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load session: {str(e)}")


@app.delete("/api/sessions/{session_id}", tags=["Sessions"])
def delete_session_by_id(session_id: str):
    """Delete a saved session by ID."""
    try:
        storage.delete_session(session_id)
        return {"status": "success", "message": f"Session {session_id} deleted."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete session: {str(e)}")


@app.post("/api/export/csv", tags=["Export"])
def export_csv_file(payload: ExportCSVRequest):
    """Generate and download a CSV containing recovered spectrum and uncertainty bands."""
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "wavelength_nm",
        "observed_noisy_flux",
        "ai_recovered_flux",
        "uncertainty_lower",
        "uncertainty_upper",
    ])

    for i in range(len(payload.wavelengths)):
        writer.writerow([
            payload.wavelengths[i],
            f"{payload.noisy_spec[i]:.4f}",
            f"{payload.recovered_spec[i]:.4f}",
            f"{payload.unc_lower[i]:.4f}",
            f"{payload.unc_upper[i]:.4f}",
        ])

    return Response(
        content=output.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=ourionspectra_recovered_spectrum.csv"},
    )


@app.post("/api/export/pdf", tags=["Export"])
def export_pdf_report(payload: ExportPDFRequest):
    """Generate and download a multi-page PDF summary report."""
    buf = io.BytesIO()
    colors = {
        "navy": NAVY,
        "blue_accent": BLUE_ACCENT,
        "gray_line": GRAY_LINE,
        "text_sub": TEXT_SUB,
    }

    try:
        export_report_pdf(
            path=buf,
            wavelengths=payload.wavelengths,
            noisy_spec=payload.noisy_spec,
            recovered_spec=payload.recovered_spec,
            unc_lower=payload.unc_lower,
            unc_upper=payload.unc_upper,
            source_label=payload.source_label,
            rmse_before=payload.rmse_before,
            rmse_after=payload.rmse_after,
            recovery_pct=payload.recovery_pct,
            features_rows=payload.features_rows,
            colors=colors,
            mae_val=payload.mae_val,
            reference_spec=payload.reference_spec,
        )
        buf.seek(0)
        return Response(
            content=buf.getvalue(),
            media_type="application/pdf",
            headers={"Content-Disposition": "attachment; filename=ourionspectra_report.pdf"},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate PDF: {str(e)}")
