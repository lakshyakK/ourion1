from __future__ import annotations
import json
from pathlib import Path
import numpy as np, torch
from scipy.ndimage import gaussian_filter1d
from .fast_recovery import SpectralPointRecoveryMLP
from .external_validation_stage11 import load_external

def metrics(a,b):
 e=np.asarray(a)-np.asarray(b); return {'rmse':float(np.sqrt(np.mean(e*e))),'mae':float(np.mean(np.abs(e)))}

def baseline(wl,n):
 if len(wl)<3:return n.copy()
 sp=max(float(np.median(np.diff(wl))),1e-4); return gaussian_filter1d(n, max(1.,0.02/sp),mode='nearest')

def feat(wl,n,sig):
 wl=np.asarray(wl,float);n=np.asarray(n,float);sig=np.asarray(sig,float);grad=np.gradient(n,wl) if len(wl)>1 else np.zeros_like(n);curv=np.gradient(grad,wl) if len(wl)>2 else np.zeros_like(n);base=baseline(wl,n); return np.column_stack([(wl-.34)/(5.17-.34),n,sig,base,n-base,grad*0.01,curv*0.0001,np.ones_like(n)])

def load_model(path):
 ck=torch.load(path,map_location='cpu',weights_only=True);m=SpectralPointRecoveryMLP();m.load_state_dict(ck['model_state']);m.eval();return m

def evaluate_target(path,model_path,seed=1616,scales=(0.5,1.,1.5,2.,2.5),n=60):
 import pandas as pd
 p=Path(path)
 if p.name=='wasp39b_reference.csv':
  df=pd.read_csv(p);wl=df.wavelength_micron.to_numpy(float);clean=df.normalized_flux.to_numpy(float);sig=df.normalized_uncertainty.to_numpy(float);v=np.isfinite(clean)&np.isfinite(sig)&(sig>0);wl,clean,sig=wl[v],clean[v],sig[v]
 elif p.name=='hatp1b_wakeford2013.csv':
  df=pd.read_csv(p);wl=df.wavelength_micron.to_numpy(float);r=df.rp_over_rstar.to_numpy(float);sr=df.rp_over_rstar_uncertainty.to_numpy(float);wl,clean,sig=wl,1-r*r,2*r*sr
 elif 'hatp26b' in p.name:
  df=pd.read_csv(p);wl=df.wavelength_micron.to_numpy(float);r=df.rp_over_rstar.to_numpy(float);sr=df.rp_over_rstar_uncertainty.to_numpy(float);wl,clean,sig=wl,1-r*r,2*r*sr
 else:
  df=pd.read_csv(p);wl=df.wavelength_micron.to_numpy(float);clean=1-df.transit_depth_percent.to_numpy(float)/100;sig=df.transit_depth_percent_uncertainty.to_numpy(float)/100
 m=load_model(model_path);rng=np.random.default_rng(seed);out=[]
 for scale in scales:
  rr=[];bb=[];nn=[];cov=[]
  for _ in range(n):
   noisy=clean+rng.normal(0,scale*sig);X=feat(wl,noisy,scale*sig)
   with torch.no_grad(): pred,ls=m(torch.from_numpy(X.astype('float32'))); pred=pred[:,0].numpy();ps=np.exp(ls[:,0].numpy())
   rr.append(metrics(noisy,clean));bb.append(metrics(baseline(wl,noisy),clean));nn.append(metrics(pred,clean));cov.append(np.mean(np.abs(pred-clean)<=ps))
  mean=lambda a,k:float(np.mean([x[k] for x in a]));out.append({'scale':scale,'raw_rmse':mean(rr,'rmse'),'baseline_rmse':mean(bb,'rmse'),'neural_rmse':mean(nn,'rmse'),'raw_mae':mean(rr,'mae'),'baseline_mae':mean(bb,'mae'),'neural_mae':mean(nn,'mae'),'coverage_1sigma':float(np.mean(cov))})
 return out

def main():
 root=Path(__file__).resolve().parents[1]; out=root/'artifacts/recovery_model/stage15';out.mkdir(parents=True,exist_ok=True);mp=root/'artifacts/recovery_model/stage14_fast/model.pt'
 results={
 'wasp39b':evaluate_target(root/'data/wasp39b/wasp39b_reference.csv',mp),
 'hatp26b':evaluate_target(root/'data/external_validation/hatp26b_wakeford2017.csv',mp),
 'wasp17b':evaluate_target(root/'data/external_validation/wasp17b_alderson2022.csv',mp),
 'hatp1b_external':evaluate_target(root/'data/external_validation/hatp1b_wakeford2013.csv',mp)
 }
 (out/'evaluation.json').write_text(json.dumps(results,indent=2));
 for k,v in results.items(): print(k, json.dumps(v,indent=2))
if __name__=='__main__': main()
