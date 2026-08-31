import tempfile
import unittest
from pathlib import Path
import numpy as np

from ourionspectra.evaluate_baselines import _moving_average_masked, _metrics

class BaselineEvaluationTests(unittest.TestCase):
    def test_masked_smoothing_preserves_gaps(self):
        x=np.array([1.,2.,np.nan,4.,5.])
        valid=np.isfinite(x)
        y=_moving_average_masked(x,valid,1)
        self.assertTrue(np.isfinite(y[0]))
        self.assertTrue(np.isnan(y[2]))
        self.assertAlmostEqual(y[1],1.5)

    def test_metrics_ignores_invalid_points(self):
        p=np.array([1.,2.,9.])
        t=np.array([1.,4.,8.])
        m=np.array([True,True,False])
        r=_metrics(p,t,m)
        self.assertEqual(r['n'],2)
        self.assertAlmostEqual(r['mae'],1.0)

if __name__=='__main__': unittest.main()
