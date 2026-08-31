import tempfile, unittest
from pathlib import Path
import numpy as np, torch
from ourionspectra.generalized_recovery import GeneralizedSpectralRecoveryNet, loss_fn
from ourionspectra.train_generalized_recovery import collate, VariableSpectrumDataset

class Stage12Tests(unittest.TestCase):
    def test_variable_length_forward(self):
        m=GeneralizedSpectralRecoveryNet();x=torch.randn(2,6,31);x[:,5]=1
        mean,ls=m(x);self.assertEqual(mean.shape,(2,1,31));self.assertEqual(ls.shape,(2,1,31))
    def test_loss_mask(self):
        m=GeneralizedSpectralRecoveryNet();x=torch.randn(1,6,20);x[:,5]=1
        mean,ls=m(x);y=torch.randn(1,1,20);mask=torch.ones_like(y);mask[:,:,8:12]=0
        a,_=loss_fn(mean,ls,y,mask);self.assertTrue(torch.isfinite(a))
    def test_collate_variable_lengths(self):
        b=[]
        for n in (12,19):
            b.append((torch.randn(6,n),torch.randn(1,n),torch.ones(1,n)))
        x,y,m=collate(b);self.assertEqual(x.shape,(2,6,19));self.assertEqual(y.shape,(2,1,19));self.assertEqual(m[0,0,-1].item(),0.0)
    def test_reference_files_present(self):
        p=Path('data/wasp39b/training/train.json');self.assertTrue(p.exists())

if __name__=='__main__': unittest.main()
