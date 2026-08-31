from __future__ import annotations
import json, random
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader, Dataset
from scipy.ndimage import gaussian_filter1d
from .generalized_recovery import GeneralizedSpectralRecoveryNet, loss_fn

class CachedDataset(Dataset):
    def __init__(self,path,augment=False,seed=1):
        self.raw=json.loads(Path(path).read_text()); self.items=[]
        for idx,s in enumerate(self.raw):
            rng=np.random.default_rng(seed+idx*7919)
            wl=np.asarray(s['wavelength'],np.float32); clean=np.asarray(s['clean_flux'],np.float32); noisy=np.asarray(s['noisy_flux'],np.float32); sigma=np.asarray(s['noise_sigma'],np.float32)
            valid=np.isfinite(wl)&np.isfinite(clean)&np.isfinite(noisy)&np.isfinite(sigma)&(sigma>0); inds=np.flatnonzero(valid)
            if augment and len(inds)>28 and rng.random()<0.65:
                keep_n=int(rng.integers(28,min(len(inds),500)+1)); bins=np.linspace(0,len(inds),min(keep_n,80)+1,dtype=int); chosen=[]
                for a,b in zip(bins[:-1],bins[1:]):
                    if b>a: chosen.append(int(inds[rng.integers(a,b)]))
                pool=np.setdiff1d(inds,np.asarray(chosen,dtype=int),assume_unique=False); rem=keep_n-len(chosen)
                if rem>0 and len(pool): chosen.extend(rng.choice(pool,size=min(rem,len(pool)),replace=False).tolist())
                inds=np.sort(np.asarray(chosen,dtype=int))
            wl=wl[inds]; clean=clean[inds]; noisy=noisy[inds]; sigma=sigma[inds]
            medsp=float(np.median(np.diff(wl))) if len(wl)>1 else .01; bw=max(medsp*5,.01); sp=max(bw/max(medsp,1e-6),1.)
            base=gaussian_filter1d(noisy,sp,mode='nearest').astype(np.float32)
            spacing=np.empty_like(wl); spacing[0]=medsp; spacing[1:]=np.diff(wl); spacing=np.clip(spacing,1e-5,1.)
            x=np.stack([(wl-.63)/(5.17-.63),spacing/.005,noisy,sigma,base,np.ones_like(wl)],0).astype(np.float32)
            self.items.append((torch.from_numpy(x),torch.from_numpy(clean[None]),torch.ones((1,len(clean)),dtype=torch.float32),s.get('target_id','unknown')))
    def __len__(self): return len(self.items)
    def __getitem__(self,i): return self.items[i]

def collate(batch):
    L=max(x.shape[-1] for x,_,_,_ in batch); xs=[];ys=[];ms=[];tg=[]
    for x,y,m,t in batch:
        p=L-x.shape[-1]; xs.append(torch.nn.functional.pad(x,(0,p)));ys.append(torch.nn.functional.pad(y,(0,p)));ms.append(torch.nn.functional.pad(m,(0,p)));tg.append(t)
    return torch.stack(xs),torch.stack(ys),torch.stack(ms),tg

def train(data_dir='data/multitarget_training',out='artifacts/recovery_model/stage14',epochs=8,batch_size=32,lr=7e-4,seed=1414):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(8);Path(out).mkdir(parents=True,exist_ok=True);dev='cuda' if torch.cuda.is_available() else 'cpu'
    tr=CachedDataset(Path(data_dir)/'train.json',True,seed);va=CachedDataset(Path(data_dir)/'validation.json',False,seed+100000)
    tl=DataLoader(tr,batch_size=batch_size,shuffle=True,collate_fn=collate,num_workers=0);vl=DataLoader(va,batch_size=batch_size,shuffle=False,collate_fn=collate,num_workers=0)
    model=GeneralizedSpectralRecoveryNet(channels=32).to(dev);opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=3e-4);best=1e99;hist=[]
    for e in range(1,epochs+1):
        model.train();tloss=tn=0
        for x,y,m,_ in tl:
            x,y,m=x.to(dev),y.to(dev),m.to(dev);pred,ls=model(x);loss,_=loss_fn(pred,ls,y,m);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();tloss+=loss.detach().item()*int(m.sum());tn+=int(m.sum())
        model.eval();vloss=vn=0
        with torch.no_grad():
            for x,y,m,_ in vl:
                x,y,m=x.to(dev),y.to(dev),m.to(dev);pred,ls=model(x);loss,_=loss_fn(pred,ls,y,m);vloss+=loss.item()*int(m.sum());vn+=int(m.sum())
        a=tloss/max(tn,1);b=vloss/max(vn,1);hist.append({'epoch':e,'train_loss':a,'validation_loss':b})
        if b<best: best=b;torch.save({'model_state':model.state_dict(),'seed':seed,'stage':14,'channels':32},Path(out)/'model.pt')
    rep={'stage':14,'model':'GeneralizedSpectralRecoveryNet','channels':32,'training_samples':len(tr),'validation_samples':len(va),'epochs':epochs,'best_validation_loss':best,'seed':seed,'training_targets':['WASP-39b','HAT-P-26b','WASP-17b'],'external_holdout':['HAT-P-1b'],'test_used_for_training':False,'history':hist}
    (Path(out)/'training_report.json').write_text(json.dumps(rep,indent=2));return rep
if __name__=='__main__': print(json.dumps(train(),indent=2))
