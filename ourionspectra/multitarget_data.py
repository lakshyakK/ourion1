"""Stage 13: multi-target reference-candidate corpus and synthetic observations.

This module uses only independently published/processed reference candidates and
creates noisy realizations from their reported uncertainties. It never labels a
published observation as ground truth and never invents wavelengths.
"""
from __future__ import annotations
import csv, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WASP39 = ROOT / 'data/wasp39b/wasp39b_reference.csv'
HAT26 = ROOT / 'data/external_validation/hatp26b_wakeford2017.csv'
WASP17 = ROOT / 'data/external_validation/wasp17b_alderson2022.csv'
OUT = ROOT / 'data/multitarget_training'

SOURCES = {
    'wasp39b': {'path': WASP39, 'kind': 'ourionspectra_reference_csv'},
    'hatp26b': {'path': HAT26, 'kind': 'published_table_rp_over_rstar'},
    'wasp17b': {'path': WASP17, 'kind': 'published_table_transit_depth_percent'},
}

def load_candidate(name: str):
    src=SOURCES[name]; p=src['path']
    if name=='wasp39b':
        import pandas as pd
        df=pd.read_csv(p)
        wl=df['wavelength_micron'].to_numpy(float)
        flux=df['normalized_flux'].to_numpy(float)
        sig=df['normalized_uncertainty'].to_numpy(float)
        valid=np.isfinite(wl)&np.isfinite(flux)&np.isfinite(sig)&(sig>0)
        return wl[valid],flux[valid],sig[valid]
    import pandas as pd
    df=pd.read_csv(p)
    if name=='hatp26b':
        wl=df.wavelength_micron.to_numpy(float); r=df.rp_over_rstar.to_numpy(float); sr=df.rp_over_rstar_uncertainty.to_numpy(float)
        # Resolve the one duplicated 0.957 micron point by retaining the WFC3 G102 measurement,
        # because it has the smaller reported uncertainty and is the higher-quality near-IR mode.
        keep=np.ones(len(df),bool)
        dup=np.where(np.isclose(wl,0.957,atol=1e-9))[0]
        if len(dup)>1:
            best=dup[np.argmin(sr[dup])]; keep[dup]=False; keep[best]=True
        wl=wl[keep]; r=r[keep]; sr=sr[keep]
        depth=r*r; depth_sig=2*r*sr
        return wl,1-depth,depth_sig
    wl=df.wavelength_micron.to_numpy(float); depth=df.transit_depth_percent.to_numpy(float)/100.0; sig=df.transit_depth_percent_uncertainty.to_numpy(float)/100.0
    return wl,1-depth,sig

def _correlated(rng,n,rho):
    if n<2: return rng.normal(size=n)
    x=rng.normal(size=n); y=np.empty(n); y[0]=x[0]
    a=float(np.exp(-1.0/rho)); b=float(np.sqrt(max(1-a*a,1e-8)))
    for i in range(1,n): y[i]=a*y[i-1]+b*x[i]
    y=(y-np.mean(y))/(np.std(y)+1e-12); return y

def generate(seed=1313, train_per_target=600, val_per_target=150, test_per_target=150):
    OUT.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(seed)
    splits={'train':[],'validation':[],'test':[]}
    scales_train=(0.45,2.4); scales_val=(0.55,2.6); scales_test=(0.35,3.0)
    for target in SOURCES:
        wl,clean,sigma=load_candidate(target)
        for split,count,lohi,offset in [('train',train_per_target,scales_train,0),('validation',val_per_target,scales_val,100000),('test',test_per_target,scales_test,200000)]:
            local=np.random.default_rng(seed+offset+sum(map(ord,target)))
            for j in range(count):
                alpha=float(local.uniform(*lohi))
                corr_frac=float(local.uniform(0,0.25))
                white=local.normal(0,alpha*sigma)
                corr=_correlated(local,len(wl),max(2.0,min(20.0,len(wl)/6))) * (corr_frac*alpha*sigma)
                noisy=clean+white+corr
                sample={'wavelength':wl.tolist(),'clean_flux':clean.tolist(),'noisy_flux':noisy.tolist(),'noise_sigma':(alpha*sigma).tolist(),'sample_id':f'{target}_{split}_{j:04d}','target_id':target,'noise_scale':alpha,'correlated_fraction':corr_frac}
                splits[split].append(sample)
    for k,v in splits.items():
        rng.shuffle(v); (OUT/f'{k}.json').write_text(json.dumps(v),encoding='utf-8')
    metadata={
      'stage':13,'targets':list(SOURCES),'samples_per_target':{'train':train_per_target,'validation':val_per_target,'test':test_per_target},
      'total_samples':sum(map(len,splits.values())),'split_sizes':{k:len(v) for k,v in splits.items()},
      'noise_method':'heteroskedastic Gaussian noise from each reference candidate uncertainty, plus zero-mean correlated AR(1)-like systematic field scaled locally by empirical uncertainty',
      'noise_scale_ranges':{'train':list(scales_train),'validation':list(scales_val),'test':list(scales_test)},
      'seed':seed,'external_holdout':'HAT-P-1b remains evaluation-only and is not included.',
      'sources':{k:{'path':str(v['path'].relative_to(ROOT)),'kind':v['kind']} for k,v in SOURCES.items()},
      'limitations':['Published transmission spectra are observational reference candidates, not analytical ground truth.','The corpus mixes instruments and reductions; this is intentional domain diversity but is not a homogeneous instrument model.','Synthetic noise is a statistical observation proxy, not a detector-level simulator.','No new atmospheric features are added.']
    }
    (OUT/'dataset_metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    return metadata

if __name__=='__main__': print(json.dumps(generate(),indent=2))
