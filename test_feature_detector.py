import unittest
from ourionspectra.model import detect_atmospheric_features
from ourionspectra.science import generate_sample_spectrum


class TestFeatureDetector(unittest.TestCase):
    def test_empty_input(self):
        self.assertEqual(detect_atmospheric_features([], []), [])

    def test_sample_spectrum_features(self):
        wl, ref, noisy = generate_sample_spectrum(0.1)
        features = detect_atmospheric_features(wl, ref, noise_level=0.1)
        self.assertIsInstance(features, list)
        self.assertGreater(len(features), 0)
        
        for f in features:
            self.assertIn("name", f)
            self.assertIn("wl_nm", f)
            self.assertIn("confidence", f)
            self.assertIn("status", f)
            self.assertIn(f["status"], {"Detected", "Tentative", "Not Detected"})
            self.assertTrue(0 <= f["confidence"] <= 100)


if __name__ == "__main__":
    unittest.main()
