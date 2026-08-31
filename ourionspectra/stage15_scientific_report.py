from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def main():
 root=Path(__file__).resolve().parents[1]; out=root/'artifacts/recovery_model/stage15';out.mkdir(parents=True,exist_ok=True)
 r=json.loads((out/'evaluation.json').read_text())
 # Aggregate target-level mean RMSE across noise regimes.
 summary={}
 for target,rows in r.items():
  summary[target]={k:float(np.mean([row[k] for row in rows])) for k in ['raw_rmse','baseline_rmse','neural_rmse','raw_mae','baseline_mae','neural_mae','coverage_1sigma']}
 targets=list(summary)
 x=np.arange(len(targets));w=.25
 fig,ax=plt.subplots(figsize=(10,5));ax.bar(x-w,[summary[t]['raw_rmse'] for t in targets],w,label='Raw');ax.bar(x,[summary[t]['baseline_rmse'] for t in targets],w,label='Classical baseline');ax.bar(x+w,[summary[t]['neural_rmse'] for t in targets],w,label='Multi-target neural');ax.set_xticks(x,targets);ax.set_ylabel('Mean RMSE');ax.set_title('Stage 15: multi-target scientific benchmark');ax.legend();fig.tight_layout();fig.savefig(out/'multitarget_rmse_benchmark.png',dpi=160);plt.close(fig)
 fig,ax=plt.subplots(figsize=(10,5));
 for t in targets:
  ax.plot([q['scale'] for q in r[t]],[q['coverage_1sigma'] for q in r[t]],marker='o',label=t)
 ax.axhline(0.6827,linestyle='--',label='Gaussian 1σ reference');ax.set_xlabel('Noise scale');ax.set_ylabel('Empirical 1σ coverage');ax.set_ylim(0,1.05);ax.set_title('Uncertainty coverage');ax.legend();fig.tight_layout();fig.savefig(out/'uncertainty_coverage.png',dpi=160);plt.close(fig)
 report={'stage':15,'summary':summary,'external_holdout':'HAT-P-1b','selection_decision':'NOT_READY_FOR_DEFAULT_INTEGRATION','reason':'The multi-target neural model does not consistently outperform the transparent classical baseline across training-domain targets or the independent HAT-P-1b target.','scientific_claim_allowed':'The model is an experimental multi-target recovery candidate; cross-target generalization is not yet established.','plots':['multitarget_rmse_benchmark.png','uncertainty_coverage.png']}
 (out/'STAGE15_REPORT.md').write_text('# Stage 15 — Final scientific validation\n\n'+json.dumps(report,indent=2),encoding='utf-8');return report
if __name__=='__main__': print(json.dumps(main(),indent=2))
