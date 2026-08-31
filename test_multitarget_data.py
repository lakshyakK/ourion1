import json, tempfile, unittest
from pathlib import Path
import numpy as np
from ourionspectra.multitarget_data import load_candidate

class TestMultiTargetData(unittest.TestCase):
    def test_all_sources_load(self):
        for name in ('wasp39b','hatp26b','wasp17b'):
            wl,flux,sig=load_candidate(name)
            self.assertGreaterEqual(len(wl),20)
            self.assertTrue(np.all(np.isfinite(wl)))
            self.assertTrue(np.all(np.isfinite(flux)))
            self.assertTrue(np.all(sig>0))
    def test_reference_not_flattened(self):
        for name in ('hatp26b','wasp17b'):
            _,flux,_=load_candidate(name)
            self.assertGreater(np.std(flux),1e-4)
    def test_wasp39_gap_not_used_as_training_point(self):
        wl,_,_=load_candidate('wasp39b')
        self.assertEqual(len(wl),883)
        self.assertFalse(np.any((wl>=3.715)&(wl<=3.830)))

if __name__=='__main__': unittest.main()
