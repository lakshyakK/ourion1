import json
import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ourionspectra.training_data import (
    ReferenceSpectrum,
    generate_dataset,
    generate_noisy_realization,
    load_reference_spectrum,
    save_dataset_and_metadata,
)


class TestTrainingDataGenerator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ref_path = "data/wasp39b/wasp39b_reference.csv"
        cls.ref = load_reference_spectrum(cls.ref_path)

    def test_reference_spectrum_loading(self):
        self.assertIsInstance(self.ref, ReferenceSpectrum)
        self.assertEqual((self.ref.n_total, self.ref.n_valid, self.ref.n_gaps), (909, 883, 26))
        self.assertTrue(np.all(np.diff(self.ref.wavelengths) > 0))
        self.assertTrue(np.all(self.ref.uncertainties[self.ref.valid_mask] > 0))

    def test_gaps_preserved_as_nan(self):
        self.assertTrue(np.all(np.isnan(self.ref.clean_flux[~self.ref.valid_mask])))
        self.assertTrue(np.all(np.isnan(self.ref.uncertainties[~self.ref.valid_mask])))
        gap_wl = self.ref.wavelengths[~self.ref.valid_mask]
        self.assertTrue(np.any((gap_wl >= 3.715) & (gap_wl <= 3.830)))

    def test_single_realization_preserves_reference_and_gaps(self):
        rng = np.random.default_rng(123)
        sample = generate_noisy_realization(self.ref, rng, noise_scale=1.0, systematic_fraction=0.2)
        valid = self.ref.valid_mask
        clean = np.asarray(sample["clean_flux"], dtype=float)
        noisy = np.asarray(sample["noisy_flux"], dtype=float)
        sigma = np.asarray(sample["noise_sigma"], dtype=float)
        self.assertTrue(np.allclose(clean[valid], self.ref.clean_flux[valid]))
        self.assertTrue(np.all(np.isnan(noisy[~valid])))
        self.assertTrue(np.all(np.isnan(sigma[~valid])))
        self.assertEqual(len(sample["wavelength"]), self.ref.n_total)

    def test_noise_scaling(self):
        # Use pure Gaussian noise so the expected variance ratio is controlled.
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(43)
        low = generate_noisy_realization(self.ref, rng1, noise_scale=0.5)
        high = generate_noisy_realization(self.ref, rng2, noise_scale=2.0)
        clean = self.ref.clean_flux[self.ref.valid_mask]
        low_res = np.asarray(low["noisy_flux"], dtype=float)[self.ref.valid_mask] - clean
        high_res = np.asarray(high["noisy_flux"], dtype=float)[self.ref.valid_mask] - clean
        ratio = np.std(high_res) / np.std(low_res)
        self.assertAlmostEqual(ratio, 4.0, delta=0.45)

    def test_uncertainty_is_empirical_and_scales_per_channel(self):
        sample = generate_noisy_realization(self.ref, np.random.default_rng(1), noise_scale=1.7)
        sigma = np.asarray(sample["noise_sigma"], dtype=float)[self.ref.valid_mask]
        expected = self.ref.uncertainties[self.ref.valid_mask] * 1.7
        self.assertTrue(np.allclose(sigma, expected))

    def test_independent_streams_are_not_near_duplicates(self):
        train, val, test, _ = generate_dataset(self.ref_path, 20, 10, 10, seed=999)
        def residual(s):
            return np.asarray(s["noisy_flux"], dtype=float)[self.ref.valid_mask] - self.ref.clean_flux[self.ref.valid_mask]
        # Compare a small cross-split matrix; independent noise should not correlate strongly.
        for a in train[:5]:
            for b in val[:5] + test[:5]:
                corr = np.corrcoef(residual(a), residual(b))[0, 1]
                self.assertLess(abs(corr), 0.25)

    def test_test_regime_is_outside_train_scale_range(self):
        train, _, test, _ = generate_dataset(self.ref_path, 100, 20, 100, seed=1234)
        train_scales = np.asarray([s["noise_scale"] for s in train])
        test_scales = np.asarray([s["noise_scale"] for s in test])
        self.assertTrue(np.all((test_scales < 0.45) | (test_scales > 2.60)))
        self.assertGreater(train_scales.min(), 0.45 - 1e-12)
        self.assertLess(train_scales.max(), 2.60 + 1e-12)

    def test_reproducibility(self):
        a = generate_dataset(self.ref_path, 5, 3, 3, seed=2026)
        b = generate_dataset(self.ref_path, 5, 3, 3, seed=2026)
        self.assertEqual(a[0], b[0])
        self.assertEqual(a[1], b[1])
        self.assertEqual(a[2], b[2])

    def test_metadata_file_saved(self):
        train, val, test, ref = generate_dataset(self.ref_path, 2, 2, 2, seed=7)
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            meta = save_dataset_and_metadata(train, val, test, ref, tmp, seed=7)
            with open(os.path.join(tmp, "dataset_metadata.json"), encoding="utf-8") as f:
                saved = json.load(f)
            self.assertEqual(saved["number_of_wavelength_points"], 909)
            self.assertTrue(saved["preservation"]["reference_flux_unchanged"])
            self.assertTrue(saved["leakage_prevention"]["test_noise_regimes_outside_train_range"])
            self.assertEqual(meta["random_seed"], 7)


if __name__ == "__main__":
    unittest.main()
