"""
Pydantic schemas for OurionSpectra REST API.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    app_name: str = "OURIONSPECTRA API"
    version: str = "1.0.0"
    model_name: str


class SpectrumSampleRequest(BaseModel):
    noise_level: float = Field(default=0.30, ge=0.0, le=1.0, description="Noise level between 0.0 and 1.0")
    wl_start: float = Field(default=900.0, description="Start wavelength in nm")
    wl_end: float = Field(default=2500.0, description="End wavelength in nm")
    wl_step: float = Field(default=22.0, description="Wavelength step in nm")


class SpectrumSampleResponse(BaseModel):
    wavelengths: List[float]
    reference_spec: List[float]
    noisy_spec: List[float]
    noise_level: float
    dataset: str
    target: str
    instrument: str


class CSVParsePreviewResponse(BaseModel):
    headers: List[str]
    preview_rows: List[List[str]]
    total_data_rows: int
    target: str
    suggested_wl_idx: int
    suggested_flux_idx: int
    suggested_unit: str
    available_units: List[str]


class CSVProcessRequest(BaseModel):
    csv_content: str
    wl_idx: int = Field(default=0, ge=0)
    flux_idx: int = Field(default=1, ge=0)
    unit: str = Field(default="nm (nanometers)")
    normalize: bool = False
    dataset_name: Optional[str] = "Uploaded Spectrum"
    target: Optional[str] = None
    instrument: Optional[str] = "User-supplied CSV"


class SpectrumDataResponse(BaseModel):
    wavelengths: List[float]
    noisy_spec: List[float]
    target: str
    dataset: str
    instrument: str
    skipped_rows: int
    warnings: List[str]


class RecoveryMetrics(BaseModel):
    rmse_before: Optional[float] = None
    rmse_after: Optional[float] = None
    mae: Optional[float] = None
    recovery_improvement: Optional[float] = None


class FeatureItem(BaseModel):
    name: str
    wl_nm: Optional[float] = None
    confidence: Optional[float] = None
    status: str



class CompositionItem(BaseModel):
    molecule: str
    log_abundance: float


class CompositionResult(BaseModel):
    available: bool
    wavelength_min_um: Optional[float] = None
    wavelength_max_um: Optional[float] = None
    model: Optional[str] = None
    parameters: List[CompositionItem] = []


class RecoveryRequest(BaseModel):
    wavelengths: List[float]
    noisy_spec: List[float]
    reference_spec: Optional[List[float]] = None
    noise_level: float = Field(default=0.30, ge=0.0, le=1.0)
    restore_percent: float = Field(default=100.0, ge=0.0, le=100.0, description="Restoration amount percentage from 0.0 to 100.0")
    dataset: Optional[str] = "Unnamed Dataset"
    target: Optional[str] = "Not specified"
    instrument: Optional[str] = "User-supplied CSV"
    save_session: bool = True



class RecoveryResponse(BaseModel):
    session_id: Optional[str] = None
    wavelengths: List[float]
    noisy_spec: List[float]
    reference_spec: Optional[List[float]] = None
    recovered_spec: List[float]
    unc_lower: List[float]
    unc_upper: List[float]
    metrics: RecoveryMetrics
    features: List[FeatureItem]
    model_name: str
    composition: Optional[CompositionResult] = None


class FeaturesRequest(BaseModel):
    wavelengths: List[float]
    recovered_spec: List[float]
    noise_level: float = Field(default=0.30, ge=0.0, le=1.0)


class FeaturesResponse(BaseModel):
    features: List[FeatureItem]


class SessionSummary(BaseModel):
    id: str
    label: str
    saved_at: str
    recovery_improvement: Optional[float] = None


class SessionDetail(BaseModel):
    id: Optional[str] = None
    saved_at: Optional[str] = None
    dataset: Optional[str] = None
    target: Optional[str] = None
    instrument: Optional[str] = None
    noise_level: Optional[float] = None
    has_reference: Optional[bool] = False
    wavelengths: List[float] = []
    noisy_spec: List[float] = []
    reference_spec: Optional[List[float]] = None
    recovered_spec: Optional[List[float]] = None
    unc_lower: Optional[List[float]] = None
    unc_upper: Optional[List[float]] = None
    features: List[Dict[str, Any]] = []
    metrics: Dict[str, Any] = {}
    metrics_reference_available: Optional[bool] = False


class ExportCSVRequest(BaseModel):
    wavelengths: List[float]
    noisy_spec: List[float]
    recovered_spec: List[float]
    unc_lower: List[float]
    unc_upper: List[float]


class ExportPDFRequest(BaseModel):
    wavelengths: List[float]
    noisy_spec: List[float]
    recovered_spec: Optional[List[float]] = None
    unc_lower: Optional[List[float]] = None
    unc_upper: Optional[List[float]] = None
    source_label: str = "Spectrum Report"
    rmse_before: Optional[float] = None
    rmse_after: Optional[float] = None
    recovery_pct: Optional[float] = None
    mae_val: Optional[float] = None
    reference_spec: Optional[List[float]] = None
    features_rows: Optional[List[List[Any]]] = None
