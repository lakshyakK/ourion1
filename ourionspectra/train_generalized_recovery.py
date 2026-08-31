"""Train Stage 12 variable-resolution, mask-aware recovery model.

Training uses only WASP-39b synthetic observations. Augmentations simulate
observation sampling/resolution loss by selecting existing observed channels;
no new wavelength or flux values are interpolated or invented.
"""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter1d
import torch
from torch.utils.data import Dataset, DataLoader
from .generalized_recovery import GeneralizedSpectralRecoveryNet, loss_fn

class VariableSpectrumDataset(Dataset):
    def __init__(self, path, augment=False, seed=42):
        self.samples=json.loads(Path(path).read_text())
        self.augment=augment; self.seed=seed
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        s=self.samples[idx]; rng=np.random.default_rng(self.seed+idx*7919)
        wl=np.asarray(s['wavelength'],np.float32); clean=np.asarray(s['clean_flux'],np.float32)
        noisy=np.asarray(s['noisy_flux'],np.float32); sigma=np.asarray(s['noise_sigma'],np.float32)
        valid=np.isfinite(wl)&np.isfinite(clean)&np.isfinite(noisy)&np.isfinite(sigma)&(sigma>0)
        # Observation-resolution augmentation: retain only existing measured points.
        inds=np.flatnonzero(valid)
        if self.augment and len(inds)>28 and rng.random()<0.75:
            keep_n=int(rng.integers(28, min(len(inds), 500)+1))
            # Stratified selection across the existing wavelength range, never interpolate.
            bins=np.linspace(0,len(inds),min(keep_n,80)+1,dtype=int)
            chosen=[]
            for a,b in zip(bins[:-1],bins[1:]):
                if b>a: chosen.append(int(inds[rng.integers(a,b)]))
            remaining=keep_n-len(chosen)
            if remaining>0:
                pool=np.setdiff1d(inds,np.array(chosen,dtype=int),assume_unique=False)
                if len(pool): chosen.extend(rng.choice(pool,size=min(remaining,len(pool)),replace=False).tolist())
            inds=np.sort(np.asarray(chosen,dtype=int))
        wl=wl[inds]; clean=clean[inds]; noisy=noisy[inds]; sigma=sigma[inds]
        # Wavelength-aware local baseline. Gaussian filtering is performed in
        # sample space using a bandwidth tied to the local median spacing. It
        # never creates values at unobserved wavelengths because only retained
        # observed channels are present in this sample.
        bw=float(np.median(np.diff(wl))) * 5.0 if len(wl)>2 else 0.02
        bw=max(bw,0.01)
        sigma_pts=max(bw/max(float(np.median(np.diff(wl))),1e-6),1.0) if len(wl)>1 else 1.0
        weights=np.ones_like(noisy)
        baseline=gaussian_filter1d(noisy, sigma_pts, mode='nearest')
        baseline=baseline.astype(np.float32)
        spacing=np.empty_like(wl); spacing[0]=np.median(np.diff(wl)) if len(wl)>1 else 0.01; spacing[1:]=np.diff(wl)
        spacing=np.clip(spacing,1e-5,1.0)
        # Common scale keeps features numerically stable across instruments/grids.
        x=np.stack([(wl-0.63)/(5.17-0.63), spacing/0.005, noisy, sigma, baseline, np.ones_like(wl)],axis=0).astype(np.float32)
        y=clean[None,:].astype(np.float32); m=np.ones_like(clean,dtype=np.float32)[None,:]
        return torch.from_numpy(x),torch.from_numpy(y),torch.from_numpy(m)

def collate(batch):
    L=max(x.shape[-1] for x,_,_ in batch); xs=[];ys=[];ms=[]
    for x,y,m in batch:
        p=L-x.shape[-1]; xs.append(torch.nn.functional.pad(x,(0,p))); ys.append(torch.nn.functional.pad(y,(0,p))); ms.append(torch.nn.functional.pad(m,(0,p)))
    return torch.stack(xs),torch.stack(ys),torch.stack(ms)

def set_seed(s): random.seed(s);np.random.seed(s);torch.manual_seed(s)

def epoch(model,loader,opt=None,device='cpu'):
    model.train(opt is not None); total=n=0
    for x,y,m in loader:
        x,y,m=x.to(device),y.to(device),m.to(device)
        mean,ls=model(x); loss,_=loss_fn(mean,ls,y,m)
        if opt:
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
        total+=float(loss)*int(m.sum());n+=int(m.sum())
    return total/max(n,1)

def train(data_dir='data/wasp39b/training',out='artifacts/recovery_model/stage12',epochs=25,batch_size=16,lr=1e-3,seed=4212):
    set_seed(seed);torch.set_num_threads(2);Path(out).mkdir(parents=True,exist_ok=True);dev='cuda' if torch.cuda.is_available() else 'cpu'
    tr=VariableSpectrumDataset(Path(data_dir)/'train.json',True,seed);va=VariableSpectrumDataset(Path(data_dir)/'val.json',False,seed+100000)
    tl=DataLoader(tr,batch_size=batch_size,shuffle=True,collate_fn=collate);vl=DataLoader(va,batch_size=batch_size,shuffle=False,collate_fn=collate)
    model=GeneralizedSpectralRecoveryNet().to(dev);opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=2e-4)
    best=float('inf');hist=[]
    for e in range(1,epochs+1):
        a=epoch(model,tl,opt,dev); b=epoch(model,vl,None,dev);hist.append({'epoch':e,'train_loss':a,'validation_loss':b})
        if b<best:
            best=b;torch.save({'model_state':model.state_dict(),'seed':seed,'stage':12},Path(out)/'model.pt')
    rep={'stage':12,'model':'GeneralizedSpectralRecoveryNet','training_samples':len(tr),'validation_samples':len(va),'epochs':epochs,'best_validation_loss':best,'seed':seed,'test_used_for_training':False,'external_targets_used_for_training':False,'augmentation':'existing-channel stratified resolution/sampling dropout; no interpolation','history':hist,'limitations':['Training clean target diversity still originates from the WASP-39b reference candidate.','External target spectra remain evaluation-only.']}
    (Path(out)/'training_report.json').write_text(json.dumps(rep,indent=2));return rep
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',default='data/wasp39b/training');p.add_argument('--out',default='artifacts/recovery_model/stage12');p.add_argument('--epochs',type=int,default=25);p.add_argument('--batch-size',type=int,default=16);p.add_argument('--lr',type=float,default=1e-3);p.add_argument('--seed',type=int,default=4212);a=p.parse_args();print(json.dumps(train(a.data,a.out,a.epochs,a.batch_size,a.lr,a.seed),indent=2))
