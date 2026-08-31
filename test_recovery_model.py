import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from ourionspectra.recovery_model import SpectralRecoveryNet, masked_gaussian_nll, recovery_loss
from ourionspectra.train_recovery import SpectrumDataset


class RecoveryModelTests(unittest.TestCase):
    def test_forward_shape(self):
        model = SpectralRecoveryNet(channels=16)
        x = torch.randn(2, 4, 909)
        mean, log_sigma = model(x)
        self.assertEqual(tuple(mean.shape), (2, 1, 909))
        self.assertEqual(tuple(log_sigma.shape), (2, 1, 909))

    def test_nll_respects_mask(self):
        mean = torch.zeros(1, 1, 4)
        target = torch.tensor([[[0.0, 100.0, 0.0, 100.0]]])
        log_sigma = torch.zeros_like(mean)
        mask = torch.tensor([[[1.0, 0.0, 1.0, 0.0]]])
        loss = masked_gaussian_nll(mean, log_sigma, target, mask)
        self.assertAlmostEqual(float(loss), 0.0, places=6)

    def test_dataset_preserves_gap_mask(self):
        root = Path(__file__).resolve().parents[1]
        ds = SpectrumDataset(root / "data/wasp39b/training/train.json")
        x, y, mask = ds[0]
        self.assertEqual(x.shape[-1], 909)
        self.assertEqual(int(mask.sum()), 883)
        self.assertTrue(torch.isfinite(x).all())


if __name__ == "__main__":
    unittest.main()


class TestStructureLoss(unittest.TestCase):
    def test_recovery_loss_is_finite(self):
        torch.manual_seed(0)
        mean = torch.rand(2,1,20)
        target = torch.rand(2,1,20)
        log_sigma = torch.zeros_like(mean) - 4.0
        mask = torch.ones_like(mean)
        loss, parts = recovery_loss(mean, log_sigma, target, mask)
        self.assertTrue(torch.isfinite(loss).item())
        self.assertEqual(set(parts), {"nll", "mse", "gradient", "curvature"})

    def test_gap_mask_does_not_break_structure_loss(self):
        mean = torch.rand(1,1,10)
        target = torch.rand(1,1,10)
        log_sigma = torch.zeros_like(mean) - 4.0
        mask = torch.ones_like(mean); mask[...,4:6] = 0
        loss, _ = recovery_loss(mean, log_sigma, target, mask)
        self.assertTrue(torch.isfinite(loss).item())
