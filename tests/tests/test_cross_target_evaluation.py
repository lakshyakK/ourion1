import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ourionspectra.cross_target_evaluation import load_external_reference, write_protocol


class TestCrossTargetEvaluation(unittest.TestCase):
    def test_protocol_is_explicitly_evaluation_only(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "protocol.json"
            result = write_protocol(p)
            self.assertFalse(result["stage"] == "training")
            self.assertIn("ground truth", result["ground_truth_language"])
            self.assertTrue(p.exists())

    def test_external_loader_requires_required_columns(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "bad.csv"
            with p.open("w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(["wavelength_micron", "normalized_flux"])
                csv.writer(f).writerow([1.0, 1.0])
            with self.assertRaises(ValueError):
                load_external_reference(p)

    def test_external_loader_preserves_nan(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "good.csv"
            with p.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["wavelength_micron", "normalized_flux", "normalized_uncertainty"])
                for i in range(25):
                    w.writerow([1.0 + i * 0.01, "" if i == 10 else 1.0, "" if i == 10 else 0.01])
            x = load_external_reference(p)
            self.assertFalse(x["valid"][10])
            self.assertTrue(np.isnan(x["flux"][10]))


if __name__ == "__main__":
    unittest.main()
