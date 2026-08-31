import unittest
import numpy as np

from ourionspectra.domain_robustness import make_domain_shifted_sample
from ourionspectra.training_data import load_reference_spectrum


class DomainRobustnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ref = load_reference_spectrum()

    def test_clean_target_is_unchanged(self):
        rng = np.random.default_rng(7)
        sample = make_domain_shifted_sample(self.ref, rng, 1.0, 0.2, 0.12)
        target = np.asarray(sample["clean_flux"], dtype=float)
        self.assertTrue(np.allclose(target[self.ref.valid_mask], self.ref.clean_flux[self.ref.valid_mask]))

    def test_gaps_remain_missing_after_domain_shift(self):
        rng = np.random.default_rng(8)
        sample = make_domain_shifted_sample(self.ref, rng, 2.0, 0.2, 0.20)
        noisy = np.asarray(sample["noisy_flux"], dtype=float)
        self.assertTrue(np.all(np.isnan(noisy[~self.ref.valid_mask])))

    def test_domain_shift_metadata_is_recorded(self):
        sample = make_domain_shifted_sample(self.ref, np.random.default_rng(9), 1.0, 0.15, 0.08)
        self.assertEqual(sample["domain_shift"]["type"], "smooth_calibration_drift")
        self.assertGreater(sample["domain_shift"]["rms_in_empirical_sigma_units"], 0)


if __name__ == "__main__":
    unittest.main()
