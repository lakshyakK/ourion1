import unittest, os
import numpy as np
import ourionspectra.model as model

class TestExperimentalInference(unittest.TestCase):
    def test_default_model_remains_baseline(self):
        self.assertFalse(model.EXPERIMENTAL_ML_ENABLED)
        wl=np.linspace(.8,1.6,20).tolist(); noisy=(1+0.01*np.sin(np.linspace(0,6,20))).tolist()
        rec,lo,hi=model.run_recovery_model(wl,noisy,0.3)
        self.assertEqual(len(rec),20); self.assertEqual(len(lo),20); self.assertEqual(len(hi),20)
    def test_model_interface_stable(self):
        wl=[1.,1.1,1.2]; noisy=[1.,.99,1.01]; out=model.run_recovery_model(wl,noisy,.3); self.assertEqual(len(out),3)

if __name__=='__main__': unittest.main()
