import json
import tempfile
import unittest
from pathlib import Path

from ourionspectra.external_validation_stage11 import load_external, evaluate


class TestStage11ExternalValidation(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1]
        self.reference = self.root / "data" / "external_validation" / "hatp1b_wakeford2013.csv"
        self.model = self.root / "artifacts" / "recovery_model" / "stage9_model" / "model.pt"

    def test_reference_conversion(self):
        wl, flux, sigma = load_external(self.reference)
        self.assertEqual(len(wl), 28)
        self.assertTrue((flux < 1.0).all())
        self.assertTrue((sigma > 0).all())
        self.assertAlmostEqual(float(wl[0]), 1.1269, places=4)
        self.assertAlmostEqual(float(wl[-1]), 1.6453, places=4)

    def test_evaluation_is_external_only(self):
        with tempfile.TemporaryDirectory() as td:
            report = evaluate(self.model, self.reference, td, seed=7, realizations_per_scale=3)
            self.assertFalse(report["training_used"])
            self.assertFalse(report["validation_used"])
            self.assertFalse(report["model_selection_used"])
            self.assertFalse(report["ground_truth"])
            self.assertEqual(report["valid_points"], 28)
            self.assertTrue(Path(report["plot"]).exists())
            self.assertTrue(Path(td, "stage11_external_validation.json").exists())
            self.assertEqual(set(report["results"].keys()), {"0.5", "1.0", "1.5", "2.0"})


if __name__ == "__main__":
    unittest.main()
