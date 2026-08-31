"""
Unit tests for ourionspectra.science — pure functions, no tkinter needed.
Run with:  python -m unittest discover -s tests
"""

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ourionspectra.science import (
    bump, true_flux, rmse, moving_average, generate_sample_spectrum,
    recover_spectrum, feature_confidence, status_from_confidence,
    normalize_series, guess_column,
)
from ourionspectra.model import run_recovery_model


class TestBumpAndTrueFlux(unittest.TestCase):
    def test_bump_peak_at_center(self):
        self.assertAlmostEqual(bump(1000, 1000, 0.5, 10), 0.5)

    def test_bump_decays_away_from_center(self):
        near = bump(1000, 1000, 0.5, 10)
        far = bump(1100, 1000, 0.5, 10)
        self.assertLess(far, near)

    def test_true_flux_is_near_one_away_from_features(self):
        # far from any absorption feature, flux should be close to 1
        self.assertGreater(true_flux(1000000), 0.99)

    def test_true_flux_dips_at_water_feature(self):
        self.assertLess(true_flux(1400), true_flux(1300))


class TestRmse(unittest.TestCase):
    def test_identical_sequences_zero_error(self):
        self.assertEqual(rmse([1, 2, 3], [1, 2, 3]), 0.0)

    def test_known_value(self):
        # errors of 1 each -> rmse = 1
        self.assertAlmostEqual(rmse([0, 0, 0], [1, 1, 1]), 1.0)

    def test_mismatched_lengths_returns_zero(self):
        self.assertEqual(rmse([1, 2], [1, 2, 3]), 0.0)

    def test_empty_inputs_returns_zero(self):
        self.assertEqual(rmse([], []), 0.0)


class TestMovingAverage(unittest.TestCase):
    def test_output_same_length(self):
        data = [1, 2, 3, 4, 5]
        self.assertEqual(len(moving_average(data)), len(data))

    def test_constant_input_unchanged(self):
        data = [5, 5, 5, 5, 5]
        result = moving_average(data, window=3)
        for v in result:
            self.assertAlmostEqual(v, 5)

    def test_smooths_a_spike(self):
        data = [1, 1, 1, 100, 1, 1, 1]
        result = moving_average(data, window=3)
        self.assertLess(result[3], 100)


class TestNormalizeSeries(unittest.TestCase):
    def test_max_becomes_one(self):
        result = normalize_series([1e-16, 2e-16, 4e-16])
        self.assertAlmostEqual(max(result), 1.0)

    def test_relative_ordering_preserved(self):
        result = normalize_series([1e-16, 2e-16, 4e-16])
        self.assertLess(result[0], result[1])
        self.assertLess(result[1], result[2])

    def test_empty_input(self):
        self.assertEqual(normalize_series([]), [])

    def test_all_zero_input_no_crash(self):
        result = normalize_series([0, 0, 0])
        self.assertEqual(result, [0, 0, 0])

    def test_negative_values_handled(self):
        result = normalize_series([-4e-16, 2e-16, 4e-16])
        self.assertAlmostEqual(max(abs(v) for v in result), 1.0)


class TestGenerateSampleSpectrum(unittest.TestCase):
    def test_matching_lengths(self):
        wl, t, n = generate_sample_spectrum(0.3)
        self.assertEqual(len(wl), len(t))
        self.assertEqual(len(wl), len(n))
        self.assertGreater(len(wl), 0)

    def test_wavelength_range_respected(self):
        wl, _, _ = generate_sample_spectrum(0.3, wl_start=1000, wl_end=1100, wl_step=10)
        self.assertGreaterEqual(wl[0], 1000)
        self.assertLessEqual(wl[-1], 1100)

    def test_higher_noise_increases_deviation(self):
        _, t_low, n_low = generate_sample_spectrum(0.05)
        _, t_high, n_high = generate_sample_spectrum(0.9)
        self.assertLess(rmse(n_low, t_low), rmse(n_high, t_high))


class TestRecoverSpectrum(unittest.TestCase):
    def test_recovery_closer_to_truth_than_noisy(self):
        wl, t, n = generate_sample_spectrum(0.4)
        rec, lo, hi = recover_spectrum(wl, t, 0.4)
        self.assertLess(rmse(rec, t), rmse(n, t))

    def test_uncertainty_band_contains_recovered_value(self):
        wl, t, n = generate_sample_spectrum(0.3)
        rec, lo, hi = recover_spectrum(wl, t, 0.3)
        for r, l, h in zip(rec, lo, hi):
            self.assertLessEqual(l, r)
            self.assertLessEqual(r, h)


class TestFeatureConfidence(unittest.TestCase):
    def test_within_bounds(self):
        for _ in range(20):
            c = feature_confidence(0.9, 0.5)
            self.assertGreaterEqual(c, 5)
            self.assertLessEqual(c, 97)

    def test_status_thresholds(self):
        self.assertEqual(status_from_confidence(80), "Detected")
        self.assertEqual(status_from_confidence(50), "Tentative")
        self.assertEqual(status_from_confidence(10), "Not Detected")


class TestGuessColumn(unittest.TestCase):
    def test_finds_wavelength_by_keyword(self):
        headers = ["spectrum_file", "planet", "wavelength_micron", "FLAM_W_m2_micron"]
        idx = guess_column(headers, ("wavelength", "wl", "lambda", "wave"))
        self.assertEqual(idx, 2)

    def test_finds_flux_by_keyword_excluding_wavelength(self):
        headers = ["spectrum_file", "planet", "wavelength_micron", "FLAM_W_m2_micron"]
        wl_idx = guess_column(headers, ("wavelength", "wl", "lambda", "wave"))
        flux_idx = guess_column(headers, ("flux", "flam", "fnu"), exclude=wl_idx)
        self.assertEqual(flux_idx, 3)

    def test_falls_back_to_first_available_column(self):
        headers = ["a", "b", "c"]
        idx = guess_column(headers, ("nonexistent_keyword",))
        self.assertEqual(idx, 0)

    def test_falls_back_excluding_the_excluded_index(self):
        headers = ["a", "b", "c"]
        idx = guess_column(headers, ("nonexistent_keyword",), exclude=0)
        self.assertEqual(idx, 1)


class TestRecoveryModelSeam(unittest.TestCase):
    """Tests for the model.py plug-in seam that a real ML model will replace."""

    def test_output_shapes_match_input(self):
        wl = [1000, 1010, 1020, 1030, 1040]
        noisy = [0.9, 0.95, 1.0, 0.92, 0.98]
        rec, lo, hi = run_recovery_model(wl, noisy)
        self.assertEqual(len(rec), len(wl))
        self.assertEqual(len(lo), len(wl))
        self.assertEqual(len(hi), len(wl))

    def test_uncertainty_band_contains_recovered_value(self):
        wl = list(range(1000, 1050, 5))
        noisy = [0.9 + 0.05 * (i % 3) for i in range(len(wl))]
        rec, lo, hi = run_recovery_model(wl, noisy)
        for r, l, h in zip(rec, lo, hi):
            self.assertLessEqual(l, r)
            self.assertLessEqual(r, h)

    def test_does_not_require_ground_truth(self):
        # This is the key contract: no `true_spec` argument at all —
        # real telescope data won't have one.
        wl = [4900.4, 4901.2, 4902.0]
        noisy = [0.94, 0.95, 1.0]
        rec, lo, hi = run_recovery_model(wl, noisy)
        self.assertEqual(len(rec), 3)

    def test_reduces_noise_relative_to_input_on_smooth_signal(self):
        # A clean smooth signal plus noise: recovered should better match
        # the smooth trend than the raw noisy points do, on average.
        wl, true, noisy = generate_sample_spectrum(0.4)
        rec, lo, hi = run_recovery_model(wl, noisy, 0.4)
        self.assertLess(rmse(rec, true), rmse(noisy, true))


if __name__ == "__main__":
    unittest.main()


class TestRepeatedSpectrumParsing(unittest.TestCase):
    def test_repeated_wavelength_blocks_use_one_complete_spectrum(self):
        from ourionspectra.parser import parse_spectrum_data
        grid = [7.0, 5.0, 3.0, 1.0]
        rows = [[str(w), str(i + 0.1 * j)] for i in range(2) for j, w in enumerate(grid)]
        wl, flux, skipped = parse_spectrum_data(rows, 0, 1, unit="µm (microns)", normalize=False)
        self.assertEqual(skipped, 0)
        self.assertEqual(len(wl), 4)
        self.assertEqual(wl, [1000.0, 3000.0, 5000.0, 7000.0])
        self.assertEqual([round(v, 6) for v in flux], [0.3, 0.2, 0.1, 0.0])
