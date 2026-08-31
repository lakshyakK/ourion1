# OurionSpectra — Exoplanet Atmospheric Spectrum Recovery

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**OurionSpectra** is an AI-powered scientific software suite and web platform designed for recovering exoplanet atmospheric transmission spectra from noisy space-telescope observations (SIH project **SICT037**).

---

## 🌟 Key Features

1. **🌐 Web Application & Interactive Dashboard**:
   - Modern browser-based interface running directly via FastAPI.
   - Live spectrum visualization with Chart.js.
   - Dynamic **Restoration Amount (%)** slider for real-time spectrum denoising and reconstruction.
   - Real-time **RMSE, MAE, and Recovery Improvement %** metrics.
   - **Atmospheric Feature Detection**: Identifies key molecular absorption bands ($H_2O$, $CO_2$, $CH_4$, $CO$, $NH_3$, $Na$, $K$) with AI confidence scores.
   - **Atmospheric Composition Neural Net**: Predicts bulk log-abundances.

2. **⚡ High-Performance REST API**:
   - 11 modular endpoints covering sample generation, CSV auto-parsing, recovery inference, feature detection, session management, and CSV/PDF report exports.
   - Interactive Swagger docs at `/docs` and ReDoc at `/redoc`.

3. **🖥️ Desktop Application**:
   - Standalone GUI built with Python Tkinter & Matplotlib.


---

## Quick Start

### Option A: Run the Desktop GUI (Windows)
1. Double-click **`run.bat`**.
2. Or from terminal:
   ```bash
   pip install -r requirements.txt
   python main.py
   ```

### Option B: Run the FastAPI Backend Server
1. Double-click **`run_server.bat`**.
2. Or from terminal:
   ```bash
   pip install -r requirements.txt
   python server.py --host 127.0.0.1 --port 8000
   ```
3. Open your browser and navigate to:
   - **Interactive Swagger Documentation**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
   - **ReDoc Documentation**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)
   - **Health Endpoint**: [http://127.0.0.1:8000/api/health](http://127.0.0.1:8000/api/health)

---

## Project Structure

```
ourionspectra/
├── main.py                     # Desktop GUI entry point
├── server.py                   # FastAPI server entry point (uvicorn runner)
├── run.bat                     # Desktop GUI Windows launcher
├── run_server.bat              # FastAPI Server Windows launcher
├── build_exe.bat               # PyInstaller executable build script
├── ourionspectra.spec              # PyInstaller configuration
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── assets/                     # Icons and logos
├── data/
│   └── sessions/               # Saved session JSON files
├── tests/
│   ├── test_science.py         # Unit tests for science & math routines
│   └── test_api.py             # Integration tests for FastAPI endpoints
└── ourionspectra/
    ├── __init__.py             # Package init
    ├── config.py               # Application configuration & theme constants
    ├── parser.py               # Shared CSV parsing and metadata extraction utilities
    ├── science.py              # Pure math: spectrum synthesis, RMSE, MAE, smoothing
    ├── model.py                # Recovery model & feature detection plug-in seam
    ├── storage.py              # Local JSON session persistence (CRUD)
    ├── report.py               # Multi-page PDF report generator
    ├── widgets.py              # Custom Tkinter UI widgets (CanvasSlider, etc.)
    ├── app.py                  # Main desktop GUI application
    └── api/                    # FastAPI backend module
        ├── __init__.py
        ├── schemas.py          # Pydantic request & response models
        └── routes.py           # API route definitions
```

---

## FastAPI REST Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/health` | Healthcheck and active model information |
| `POST` | `/api/sample` | Generate synthetic near-IR transit spectrum with realistic noise |
| `POST` | `/api/spectrum/preview-file` | Upload and preview CSV files with auto-detected columns and metadata |
| `POST` | `/api/spectrum/process-csv` | Parse raw CSV content into wavelength and flux vectors |
| `POST` | `/api/spectrum/recover` | Run spectrum recovery model, compute metrics, and return uncertainty |
| `POST` | `/api/spectrum/features` | Detect atmospheric features/molecules from spectrum |
| `GET` | `/api/sessions` | List all saved recovery sessions |
| `GET` | `/api/sessions/{id}` | Retrieve full details of a saved session |
| `DELETE` | `/api/sessions/{id}` | Delete a saved session |
| `POST` | `/api/export/csv` | Export recovered spectrum data as a downloadable CSV |
| `POST` | `/api/export/pdf` | Generate and download a multi-page PDF summary report |

---

## Running the Automated Test Suite

Run the full test suite (science unit tests + API integration tests):

```bash
python -m unittest discover -s tests -v
```

---

## Building Standalone Desktop Executable (.exe)

1. Double-click **`build_exe.bat`**.
2. The standalone build will be generated in `dist\OurionSpectra\OurionSpectra.exe`.
3. The executable runs without requiring Python on target machines.

---

## Customizing the AI Model

To integrate your own deep learning / ML model:
1. Open `ourionspectra/model.py`.
2. Implement your model inference inside `run_recovery_model(wavelengths, noisy_flux, noise_level)`.
3. Implement molecular detection inside `detect_atmospheric_features(wavelengths, recovered_flux, noise_level)`.
4. Both the Desktop GUI and the FastAPI backend will automatically use the updated model without any further changes required.

## Scientific ML Status

Stages 13–17 added a multi-target reference-candidate corpus, an experimental multi-target recovery model, independent HAT-P-1b validation, uncertainty diagnostics, and a safe opt-in ML inference seam. The neural model is **not** the default recovery engine because it does not consistently outperform the transparent baseline across targets. Set `OURIONSPECTRA_ENABLE_EXPERIMENTAL_ML=1` only for controlled experiments.

No atmospheric feature detections are fabricated. Published spectra are treated as reference candidates, not ground truth.
