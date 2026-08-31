"""Stage 12 evaluation on held-out WASP-39b and independent HAT-P-1b stress test."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, torch
from scipy.ndimage import gaussian_filter1d
from .generalized_recovery import GeneralizedSpectralRecoveryNet
from .train_generalized_recovery import VariableSpectrumDataset, collate
from .external_validation_stage11 import load_external
from .evaluate_baselines import _metrics, _moving_average_masked

def _features(wl,noisy,sigma):
    valid=np.isfinite(wl)&np.isfinite(noisy)&np.isfinite(sigma)&(sigma>0)
    bw=max(float(np.median(np.diff(wl))*5.0) if len(wl)>2 else 0.02,0.01)
    sigma_pts=max(bw/max(float(np.median(np.diff(wl))),1e-6),1.0) if len(wl)>1 else 1.0
    base=gaussian_filter1d(noisy,sigma_pts,mode='nearest').astype(float)
    spacing=np.empty_like(wl);spacing[0]=np.median(np.diff(wl)) if len(wl)>1 else .01;spacing[1:]=np.diff(wl)
    x=np.stack([(wl-.63)/(5.17-.63),spacing/.005,noisy,sigma,base,valid.astype(float)],0).astype(np.float32)
    return x,base,valid

def eval_external(model_path, reference_path, outdir, seed=731):
    out=Path(outdir);out.mkdir(parents=True,exist_ok=True)
    wl,clean,sigma=load_external(reference_path); model=GeneralizedSpectralRecoveryNet();ck=torch.load(model_path,map_location='cpu',weights_only=True);model.load_state_dict(ck['model_state']);model.eval();rng=np.random.default_rng(seed)
    rows=[]
    for scale in [0.5,1.,1.5,2.]:
        vals=[]
        for _ in range(100):
            noisy=clean+rng.normal(0,scale*sigma);x,base,valid=_features(wl,noisy,scale*sigma)
            with torch.no_grad(): pred,ls=model(torch.from_numpy(x[None]));pred=pred[0,0].numpy()
            vals.append((_metrics(noisy,clean,valid),_metrics(base,clean,valid),_metrics(pred,clean,valid)))
        avg=lambda k,j: float(np.mean([r[j][k] for r in vals]))
        rows.append({'scale':scale,'raw_rmse':avg('rmse',0),'baseline_rmse':avg('rmse',1),'neural_rmse':avg('rmse',2),'raw_mae':avg('mae',0),'baseline_mae':avg('mae',1),'neural_mae':avg('mae',2)})
    rep={'stage':12,'target':'HAT-P-1b','external_evaluation_only':True,'results':rows,'limitations':['No external target used for training or model selection.','The independent spectrum has a much coarser grid and different uncertainty scale than WASP-39b.','This is a synthetic-noise stress test of a published reference candidate, not raw time-series recovery.']}
    (out/'external_evaluation.json').write_text(json.dumps(rep,indent=2));return rep

def eval_wasp(model_path,test_path,outdir):
    out=Path(outdir);out.mkdir(parents=True,exist_ok=True);ds=VariableSpectrumDataset(test_path,False,9999);model=GeneralizedSpectralRecoveryNet();ck=torch.load(model_path,map_location='cpu',weights_only=True);model.load_state_dict(ck['model_state']);model.eval();sq=ab=rsq=rab=0;n=0
    for i in range(len(ds)):
        x,y,m=ds[i];
        with torch.no_grad(): pred,_=model(x[None])
        valid=m.bool();e=(pred[0,0]-y[0])[valid[0]];r=(x[2]-y[0])[valid[0]];sq+=float((e**2).sum());ab+=float(e.abs().sum());rsq+=float((r**2).sum());rab+=float(r.abs().sum());n+=int(valid.sum())
    rep={'test_samples':len(ds),'rmse':float(np.sqrt(sq/n)),'mae':float(ab/n),'raw_rmse':float(np.sqrt(rsq/n)),'raw_mae':float(rab/n),'test_used_for_training':False}
    (out/'wasp_test_evaluation.json').write_text(json.dumps(rep,indent=2));return rep
