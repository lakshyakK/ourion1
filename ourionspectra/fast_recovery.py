from __future__ import annotations
import json, random
from pathlib import Path
import numpy as np, torch
from torch import nn

class SpectralPointRecoveryMLP(nn.Module):
    def __init__(self, hidden=96):
        super().__init__(); self.net=nn.Sequential(nn.Linear(8,hidden),nn.GELU(),nn.Linear(hidden,hidden),nn.GELU(),nn.Linear(hidden,2))
    def forward(self,x):
        out=self.net(x); sigma=x[:,2:3].clamp_min(1e-5); baseline=x[:,3:4]
        correction=0.02*torch.tanh(out[:,:1]); mean=baseline+correction
        log_sigma=torch.clamp(torch.log(sigma)+0.5*torch.tanh(out[:,1:2]),-9,-2.5)
        return mean,log_sigma

def build_arrays(json_path, augment=False, seed=1, points_per_target=60000):
    rows=json.loads(Path(json_path).read_text()); by={}
    for i,s in enumerate(rows):
        target=s.get('target_id','unknown'); rng=np.random.default_rng(seed+i*17); wl=np.asarray(s['wavelength'],float); y=np.asarray(s['clean_flux'],float); n=np.asarray(s['noisy_flux'],float); sig=np.asarray(s['noise_sigma'],float); valid=np.isfinite(wl)&np.isfinite(y)&np.isfinite(n)&np.isfinite(sig)&(sig>0); idx=np.flatnonzero(valid)
        if augment and len(idx)>40 and rng.random()<.7: idx=np.sort(rng.choice(idx,size=int(rng.integers(40,len(idx)+1)),replace=False))
        wl,n,y,sig=wl[idx],n[idx],y[idx],sig[idx]
        if len(wl)>1: grad=np.gradient(n,wl); curv=np.gradient(grad,wl)
        else: grad=np.zeros_like(n);curv=np.zeros_like(n)
        base=np.empty_like(n); sp=max(np.median(np.diff(wl)) if len(wl)>1 else .01,1e-3)
        for j in range(len(n)):
            lo=max(0,j-4);hi=min(len(n),j+5);w=np.exp(-0.5*((wl[lo:hi]-wl[j])/sp)**2);base[j]=np.sum(w*n[lo:hi])/np.sum(w)
        feats=np.column_stack([(wl-.34)/(5.17-.34),n,sig,base,n-base,grad*0.01,curv*0.0001,np.ones_like(n)])
        by.setdefault(target,[]).append((feats,y))
    X=[];Y=[];rng=np.random.default_rng(seed+999)
    for target,parts in by.items():
        fx=np.concatenate([a for a,_ in parts]); fy=np.concatenate([b for _,b in parts]);
        if len(fx)>=points_per_target: sel=rng.choice(len(fx),points_per_target,replace=False)
        else: sel=rng.choice(len(fx),points_per_target,replace=True)
        X.append(fx[sel]);Y.append(fy[sel])
    return np.concatenate(X),np.concatenate(Y)

def train(train_json,val_json,out='artifacts/recovery_model/stage14_fast',epochs=15,seed=1515):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);Path(out).mkdir(parents=True,exist_ok=True)
    X,y=build_arrays(train_json,True,seed);Xv,yv=build_arrays(val_json,False,seed+10000)
    xt=torch.from_numpy(X.astype('float32'));yt=torch.from_numpy(y[:,None].astype('float32'));xv=torch.from_numpy(Xv.astype('float32'));yv=torch.from_numpy(yv[:,None].astype('float32'))
    model=SpectralPointRecoveryMLP();opt=torch.optim.AdamW(model.parameters(),lr=2e-3,weight_decay=2e-4);best=1e99;hist=[]
    for e in range(1,epochs+1):
        model.train();perm=torch.randperm(len(xt));total=0
        for start in range(0,len(xt),16384):
            ix=perm[start:start+16384];pred,_=model(xt[ix]);loss=((pred-yt[ix])**2).mean();opt.zero_grad();loss.backward();opt.step();total+=loss.item()*len(ix)
        model.eval();
        with torch.no_grad():pred,_=model(xv);vl=((pred-yv)**2).mean().item()
        hist.append({'epoch':e,'train_mse':total/len(xt),'validation_mse':vl})
        if vl<best:best=vl;torch.save({'model_state':model.state_dict(),'seed':seed,'stage':14},Path(out)/'model.pt')
    rep={'stage':14,'model':'SpectralPointRecoveryMLP','objective':'MSE reconstruction of clean reference candidate from noisy spectrum plus baseline','train_points':len(X),'validation_points':len(Xv),'epochs':epochs,'best_validation_mse':best,'training_targets':['WASP-39b','HAT-P-26b','WASP-17b'],'external_holdout':['HAT-P-1b'],'history':hist}
    (Path(out)/'training_report.json').write_text(json.dumps(rep,indent=2));return rep
if __name__=='__main__': print(json.dumps(train('data/multitarget_training/train.json','data/multitarget_training/validation.json'),indent=2))
