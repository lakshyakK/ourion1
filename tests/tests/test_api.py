"""
Unit and integration tests for the OurionSpectra FastAPI backend.
Run with: python -m unittest discover -s tests -v
"""

import io
import os
import sys
import unittest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ourionspectra.api.routes import app
from ourionspectra import storage


class TestOurionSpectraAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health_check(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["app_name"], "OURIONSPECTRA API")
        self.assertIn("model_name", data)

    def test_sample_generation_default(self):
        response = self.client.post("/api/sample", json={})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("wavelengths", data)
        self.assertIn("noisy_spec", data)
        self.assertIn("reference_spec", data)
        self.assertEqual(len(data["wavelengths"]), len(data["noisy_spec"]))
        self.assertEqual(len(data["wavelengths"]), len(data["reference_spec"]))
        self.assertGreater(len(data["wavelengths"]), 0)

    def test_sample_generation_custom_parameters(self):
        payload = {
            "noise_level": 0.5,
            "wl_start": 1000.0,
            "wl_end": 1500.0,
            "wl_step": 50.0,
        }
        response = self.client.post("/api/sample", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertAlmostEqual(data["noise_level"], 0.5)
        self.assertEqual(data["wavelengths"][0], 1000.0)
        self.assertEqual(data["wavelengths"][-1], 1500.0)

    def test_preview_csv_file(self):
        csv_content = (
            "# Target: WASP-121 b\n"
            "wavelength_micron,flux_value\n"
            "1.0,0.95\n"
            "1.2,0.91\n"
            "1.4,0.85\n"
            "1.6,0.93\n"
        )
        file_tuple = ("test_spectrum.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")
        response = self.client.post("/api/spectrum/preview-file", files={"file": file_tuple})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["target"], "WASP-121 b")
        self.assertEqual(data["headers"], ["wavelength_micron", "flux_value"])
        self.assertEqual(data["total_data_rows"], 4)
        self.assertEqual(data["suggested_wl_idx"], 0)
        self.assertEqual(data["suggested_flux_idx"], 1)

    def test_process_csv_content(self):
        csv_content = (
            "wavelength,flux\n"
            "1000,0.98\n"
            "1100,0.95\n"
            "1200,0.92\n"
            "1300,0.89\n"
        )
        payload = {
            "csv_content": csv_content,
            "wl_idx": 0,
            "flux_idx": 1,
            "unit": "nm (nanometers)",
            "normalize": False,
            "dataset_name": "Test dataset",
        }
        response = self.client.post("/api/spectrum/process-csv", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["dataset"], "Test dataset")
        self.assertEqual(len(data["wavelengths"]), 4)
        self.assertEqual(len(data["noisy_spec"]), 4)
        self.assertEqual(data["wavelengths"], [1000.0, 1100.0, 1200.0, 1300.0])

    def test_spectrum_recovery_with_reference(self):
        sample_resp = self.client.post("/api/sample", json={"noise_level": 0.3})
        sample_data = sample_resp.json()

        payload = {
            "wavelengths": sample_data["wavelengths"],
            "noisy_spec": sample_data["noisy_spec"],
            "reference_spec": sample_data["reference_spec"],
            "noise_level": 0.3,
            "dataset": "Sample Run",
            "target": "Simulated Exoplanet",
            "instrument": "NIR Spec",
            "save_session": True,
        }
        response = self.client.post("/api/spectrum/recover", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertEqual(len(data["recovered_spec"]), len(payload["wavelengths"]))
        self.assertEqual(len(data["unc_lower"]), len(payload["wavelengths"]))
        self.assertEqual(len(data["unc_upper"]), len(payload["wavelengths"]))
        self.assertIsNotNone(data["session_id"])
        self.assertIsNotNone(data["metrics"]["rmse_before"])
        self.assertIsNotNone(data["metrics"]["rmse_after"])
        self.assertIsNotNone(data["metrics"]["mae"])
        self.assertIsNotNone(data["metrics"]["recovery_improvement"])

        # Check saved session cleanup
        session_id = data["session_id"]
        del_resp = self.client.delete(f"/api/sessions/{session_id}")
        self.assertEqual(del_resp.status_code, 200)

    def test_spectrum_recovery_without_reference(self):
        payload = {
            "wavelengths": [1000.0, 1100.0, 1200.0, 1300.0, 1400.0],
            "noisy_spec": [0.99, 0.94, 0.91, 0.88, 0.93],
            "reference_spec": None,
            "noise_level": 0.3,
            "save_session": False,
        }
        response = self.client.post("/api/spectrum/recover", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(len(data["recovered_spec"]), 5)
        self.assertIsNone(data["metrics"]["rmse_before"])
        self.assertIsNone(data["metrics"]["recovery_improvement"])

    def test_features_detection_endpoint(self):
        payload = {
            "wavelengths": [1000.0, 1100.0, 1200.0, 1300.0, 1400.0],
            "recovered_spec": [0.99, 0.95, 0.92, 0.89, 0.94],
            "noise_level": 0.3,
        }
        response = self.client.post("/api/spectrum/features", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsInstance(data["features"], list)

    def test_session_lifecycle(self):
        # Create a test session directly or through endpoint
        session_payload = {
            "wavelengths": [1000.0, 1100.0, 1200.0],
            "noisy_spec": [0.95, 0.92, 0.90],
            "reference_spec": [1.0, 0.95, 0.92],
            "noise_level": 0.2,
            "dataset": "Temporary Test Session",
            "target": "Test Planet",
            "save_session": True,
        }
        recover_resp = self.client.post("/api/spectrum/recover", json=session_payload)
        self.assertEqual(recover_resp.status_code, 200)
        session_id = recover_resp.json()["session_id"]
        self.assertIsNotNone(session_id)

        # List sessions
        list_resp = self.client.get("/api/sessions")
        self.assertEqual(list_resp.status_code, 200)
        session_ids = [s["id"] for s in list_resp.json()]
        self.assertIn(session_id, session_ids)

        # Get session detail
        get_resp = self.client.get(f"/api/sessions/{session_id}")
        self.assertEqual(get_resp.status_code, 200)
        self.assertEqual(get_resp.json()["id"], session_id)
        self.assertEqual(get_resp.json()["dataset"], "Temporary Test Session")

        # Delete session
        del_resp = self.client.delete(f"/api/sessions/{session_id}")
        self.assertEqual(del_resp.status_code, 200)

        # Confirm 404 after deletion
        get_after_del = self.client.get(f"/api/sessions/{session_id}")
        self.assertEqual(get_after_del.status_code, 404)

    def test_export_csv(self):
        payload = {
            "wavelengths": [1000.0, 1100.0],
            "noisy_spec": [0.95, 0.91],
            "recovered_spec": [0.96, 0.92],
            "unc_lower": [0.94, 0.90],
            "unc_upper": [0.98, 0.94],
        }
        response = self.client.post("/api/export/csv", json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/csv", response.headers["content-type"])
        self.assertIn("wavelength_nm,observed_noisy_flux,ai_recovered_flux", response.text)

    def test_export_pdf(self):
        payload = {
            "wavelengths": [1000.0, 1100.0, 1200.0],
            "noisy_spec": [0.95, 0.91, 0.88],
            "recovered_spec": [0.96, 0.92, 0.89],
            "unc_lower": [0.94, 0.90, 0.87],
            "unc_upper": [0.98, 0.94, 0.91],
            "source_label": "Test Exoplanet",
            "rmse_before": 0.05,
            "rmse_after": 0.02,
            "recovery_pct": 60.0,
            "features_rows": [["H2O", "1400 nm", "Detected", "85%"]],
        }
        response = self.client.post("/api/export/pdf", json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "application/pdf")
        self.assertGreater(len(response.content), 1000)
        self.assertTrue(response.content.startswith(b"%PDF"))


if __name__ == "__main__":
    unittest.main()
