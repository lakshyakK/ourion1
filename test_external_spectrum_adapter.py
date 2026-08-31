import tempfile
import unittest
from pathlib import Path

import numpy as np

from ourionspectra.external_spectrum_adapter import load_published_transmission, write_ourionspectra_csv


class ExternalSpectrumAdapterTests(unittest.TestCase):
    def _write(self, text):
        f = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False)
        f.write(text)
        f.close()
        self.addCleanup(lambda: Path(f.name).unlink(missing_ok=True))
        return f.name

    def test_loads_four_column_archive_format(self):
        path = self._write("\n".join(f"{0.6+i*0.01:.4f} 0.0001 {0.01+i*1e-5:.8f} 0.0002" for i in range(25)))
        s = load_published_transmission(path)
        self.assertEqual(len(s["wavelength_micron"]), 25)
        self.assertTrue(np.all(np.diff(s["wavelength_micron"]) > 0))
        self.assertTrue(np.all(s["normalized_uncertainty"] > 0))

    def test_rejects_too_few_points(self):
        path = self._write("\n".join("0.6 0.001 0.01 0.001" for _ in range(5)))
        with self.assertRaises(ValueError):
            load_published_transmission(path)

    def test_rejects_duplicate_wavelengths(self):
        rows = [f"0.6 0.001 {0.01+i*1e-5:.8f} 0.001" for i in range(25)]
        rows[-1] = rows[0]
        path = self._write("\n".join(rows))
        with self.assertRaises(ValueError):
            load_published_transmission(path)

    def test_writer_uses_evaluation_columns_only(self):
        path = self._write("\n".join(f"{0.6+i*0.01:.4f} 0.0001 {0.01+i*1e-5:.8f} 0.0002" for i in range(25)))
        s = load_published_transmission(path)
        with tempfile.TemporaryDirectory() as d:
            out = write_ourionspectra_csv(s, Path(d) / "external.csv")
            lines = out.read_text().splitlines()
            self.assertEqual(lines[0], "wavelength_micron,normalized_flux,normalized_uncertainty")
            self.assertEqual(len(lines), 26)


if __name__ == "__main__":
    unittest.main()
