"""Stage 7 blinded benchmark: high-noise recovery and uncertainty calibration.

The benchmark is evaluation-only. It never alters the clean reference target,
does not train on the held-out test set, and compares the hybrid model against
raw observations and transparent moving-average baselines.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from .recovery_model import SpectralRecoveryNet
from .train_recovery import SpectrumDataset
from .evaluate_baselines import _moving_average_masked, _metrics
from .domain_robustness import make_domain_shifted_sample
from .training_data import load_reference_spectrum


def _predict(model, sample):
    wl=np.asarray(sample['wavelength'],dtype=np.float32); noisy=np.asarray(sample['noisy_flux'],dtype=np.float32)
    sigma=np.asarray(sample['noise_sigma'],dtype=np.float32); clean=np.asarray(sample['clean_flux'],dtype=np.float32)
    valid=np.isfinite(clean)&np.isfinite(noisy)&np.isfinite(sigma)
    baseline=_moving_average_masked(noisy,valid,5)
    x=np.stack([(wl-0.63)/(5.17-0.63),np.nan_to_num(noisy),np.nan_to_num(sigma),np.nan_to_num(baseline)],axis=0).astype(np.float32)
    with torch.no_grad(): mean,log_sigma=model(torch.from_numpy(x[None]))
    pred=mean[0,0].numpy(); pred[~valid]=np.nan
    ps=np.exp(log_sigma[0,0].numpy()); ps[~valid]=np.nan
    return pred,ps,valid


def coverage(pred,sigma,target,valid,k=1.0):
    z=np.abs(pred-target)/(np.maximum(sigma,1e-6))
    z=z[valid]
    return float(np.mean(z<=k))


def evaluate(model_path='artifacts/recovery_model/model.pt', test_path='data/wasp39b/training/test.json', output='artifacts/recovery_model/stage7_benchmark.json'):
    model=SpectralRecoveryNet(channels=32)
    ckpt=torch.load(model_path,map_location='cpu',weights_only=True)
    model.load_state_dict(ckpt['model_state']); model.eval()
    ds=SpectrumDataset(test_path)
    methods={k:[] for k in ['raw','ma5','ma11','neural']}
    uncertainty=[]
    for sample in ds.samples:
        pred,sig,valid=_predict(model,sample)
        clean=np.asarray(sample['clean_flux'],float); noisy=np.asarray(sample['noisy_flux'],float)
        methods['raw'].append(_metrics(noisy,clean,valid)); methods['ma5'].append(_metrics(_moving_average_masked(noisy,valid,2),clean,valid)); methods['ma11'].append(_metrics(_moving_average_masked(noisy,valid,5),clean,valid)); methods['neural'].append(_metrics(pred,clean,valid))
        uncertainty.append((pred,sig,clean,valid))
    def agg(rows):
        n=sum(r['n'] for r in rows); return {'rmse':float(np.sqrt(sum(r['rmse']**2*r['n'] for r in rows)/n)),'mae':float(sum(r['mae']*r['n'] for r in rows)/n),'valid_points':n}
    test_result={k:agg(v) for k,v in methods.items()}
    cov={f'{k}sigma':float(np.mean([coverage(*u,k=k) for u in uncertainty])) for k in [1,2]}
    ref=load_reference_spectrum('data/wasp39b/wasp39b_reference.csv'); rng=np.random.default_rng(2027)
    regimes=[
      {'name':'high_noise_short','noise_scale':2.8,'drift':0.20,'length':0.04},
      {'name':'high_noise_long','noise_scale':2.8,'drift':0.20,'length':0.20},
      {'name':'very_high_noise_short','noise_scale':3.2,'drift':0.25,'length':0.05},
      {'name':'very_high_noise_long','noise_scale':3.2,'drift':0.25,'length':0.20},
      {'name':'unseen_low_noise','noise_scale':0.30,'drift':0.25,'length':0.12},
    ]
    rows=[]
    for cfg in regimes:
        s=make_domain_shifted_sample(ref,rng,cfg['noise_scale'],cfg['drift'],cfg['length'])
        pred,sig,valid=_predict(model,s); clean=np.asarray(s['clean_flux'],float); noisy=np.asarray(s['noisy_flux'],float)
        rows.append({**cfg,'raw':_metrics(noisy,clean,valid),'ma11':_metrics(_moving_average_masked(noisy,valid,5),clean,valid),'neural':_metrics(pred,clean,valid),'coverage_1sigma':coverage(pred,sig,clean,valid,1.0),'coverage_2sigma':coverage(pred,sig,clean,valid,2.0)})
    result={'stage':'7_high_noise_and_uncertainty','test_set_used_only_for_evaluation':True,'test_samples':len(ds),'held_out_test':test_result,'test_uncertainty_coverage':cov,'domain_shift_benchmark':rows,'scientific_note':'The hybrid network is explicitly conditioned on a transparent moving-average baseline. This is a fair hybrid benchmark, not evidence of cross-planet generalization. The clean target remains the original WASP-39b reference candidate and is never modified.','integration_ready':False}
    Path(output).write_text(json.dumps(result,indent=2),encoding='utf-8'); return result

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser(); p.add_argument('--model',default='artifacts/recovery_model/model.pt'); p.add_argument('--output',default='artifacts/recovery_model/stage7_benchmark.json'); a=p.parse_args(); print(json.dumps(evaluate(a.model,output=a.output),indent=2))
