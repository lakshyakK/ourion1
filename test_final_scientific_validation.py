import json, tempfile, unittest
from pathlib import Path

from ourionspectra.final_scientific_validation import _shape_errors, _bootstrap
import numpy as np

class FinalValidationTests(unittest.TestCase):
    def test_shape_error_zero_for_identical_signal(self):
        x=np.array([1.,1.1,1.05,1.2]); m=np.ones(4,dtype=bool)
        r=_shape_errors(x,x,m)
        self.assertAlmostEqual(r['gradient_rmse'],0.0)
        self.assertAlmostEqual(r['curvature_rmse'],0.0)
    def test_bootstrap_is_reproducible(self):
        a=_bootstrap(np.array([1.,2.,3.]), np.random.default_rng(7), n_boot=100)
        b=_bootstrap(np.array([1.,2.,3.]), np.random.default_rng(7), n_boot=100)
        self.assertEqual(a,b)
    def test_bootstrap_has_ordered_ci(self):
        r=_bootstrap(np.array([1.,2.,3.,4.]), np.random.default_rng(1), n_boot=100)
        self.assertLessEqual(r['ci95'][0],r['estimate'])
        self.assertLessEqual(r['estimate'],r['ci95'][1])

if __name__=='__main__': unittest.main()
