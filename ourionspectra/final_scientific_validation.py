"""Final pre-integration scientific validation for OurionSpectra.

This module freezes the current Stage-7 model and evaluates it without changing
training data, model weights, GUI, or API behavior. The held-out test set is
used only for final evaluation. No atmospheric features are inferred here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .recovery_model import SpectralRecoveryNet
from .train_recovery import SpectrumDataset, masked_moving_average


def _metrics(pred, clean, valid):
    e = np.asarray(pred)[valid] - np.asarray(clean)[valid]
    return {"rmse": float(np.sqrt(np.mean(e * e))), "mae": float(np.mean(np.abs(e))), "n": int(e.size)}


def _shape_errors(pred, clean, valid):
    p = np.asarray(pred); y = np.asarray(clean); m = np.asarray(valid)
    pair = m[1:] & m[:-1]
    trip = m[2:] & m[1:-1] & m[:-2]
    pg = p[1:] - p[:-1]; yg = y[1:] - y[:-1]
    pc = p[2:] - 2*p[1:-1] + p[:-2]; yc = y[2:] - 2*y[1:-1] + y[:-2]
    return {
        "gradient_rmse": float(np.sqrt(np.mean((pg[pair]-yg[pair])**2))),
        "curvature_rmse": float(np.sqrt(np.mean((pc[trip]-yc[trip])**2))),
    }


def _predict(model, sample):
    wl = np.asarray(sample["wavelength"], np.float32)
    noisy = np.asarray(sample["noisy_flux"], np.float32)
    sigma = np.asarray(sample["noise_sigma"], np.float32)
    clean = np.asarray(sample["clean_flux"], np.float32)
    valid = np.isfinite(clean) & np.isfinite(noisy) & np.isfinite(sigma)
    wnorm = (wl - 0.63) / (5.17 - 0.63)
    base = masked_moving_average(noisy, valid, radius=5)
    x = np.stack([np.nan_to_num(wnorm), np.nan_to_num(noisy), np.nan_to_num(sigma), np.nan_to_num(base)], axis=0).astype(np.float32)
    with torch.no_grad():
        mean, logs = model(torch.from_numpy(x[None]))
    pred = mean[0, 0].numpy(); ps = np.exp(logs[0, 0].numpy())
    pred[~valid] = np.nan; ps[~valid] = np.nan
    return pred, ps, valid, clean, noisy, base


def _bootstrap(values, rng, n_boot=1000):
    values = np.asarray(values, float)
    if len(values) < 2:
        return {"estimate": float(values.mean()), "ci95": [float(values.mean()), float(values.mean())]}
    means = np.empty(n_boot)
    for i in range(n_boot):
        means[i] = values[rng.integers(0, len(values), len(values))].mean()
    return {"estimate": float(values.mean()), "ci95": [float(np.quantile(means, .025)), float(np.quantile(means, .975))]}


def validate(model_path="artifacts/recovery_model/model.pt", test_path="data/wasp39b/training/test.json", output="artifacts/recovery_model/stage9_final_validation.json", seed=9031):
    model = SpectralRecoveryNet(channels=32)
    ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state"]); model.eval()
    ds = SpectrumDataset(test_path)
    rng = np.random.default_rng(seed)

    sample_rows=[]
    method_metrics={k:[] for k in ("raw","ma11","neural")}
    shape={k:[] for k in ("raw","ma11","neural")}
    cover=[]
    for s in ds.samples:
        pred, ps, valid, clean, noisy, base = _predict(model, s)
        ma=base
        for k,v in (("raw",noisy),("ma11",ma),("neural",pred)):
            method_metrics[k].append(_metrics(v,clean,valid))
            shape[k].append(_shape_errors(v,clean,valid))
        z=np.abs(pred[valid]-clean[valid])/np.maximum(ps[valid],1e-8)
        cover.append((float(np.mean(z<=1)), float(np.mean(z<=2))))
        sample_rows.append({"sample_id":s.get("sample_id"),"noise_scale":float(s.get("noise_scale",np.nan)),"neural_rmse":method_metrics["neural"][-1]["rmse"],"ma11_rmse":method_metrics["ma11"][-1]["rmse"]})

    def aggregate(rows):
        n=sum(r["n"] for r in rows)
        return {"rmse":float(np.sqrt(sum(r["rmse"]**2*r["n"] for r in rows)/n)),"mae":float(sum(r["mae"]*r["n"] for r in rows)/n),"valid_points":n}
    aggregate_metrics={k:aggregate(v) for k,v in method_metrics.items()}
    neural_advantage=float(1-aggregate_metrics["neural"]["rmse"]/aggregate_metrics["ma11"]["rmse"])
    raw_advantage=float(1-aggregate_metrics["neural"]["rmse"]/aggregate_metrics["raw"]["rmse"])

    sample_delta=np.array([r["ma11_rmse"]-r["neural_rmse"] for r in sample_rows])
    bootstrap=_bootstrap(sample_delta,rng)
    shape_agg={k:{"gradient_rmse":float(np.mean([r["gradient_rmse"] for r in v])),"curvature_rmse":float(np.mean([r["curvature_rmse"] for r in v]))} for k,v in shape.items()}
    coverage={"1sigma":float(np.mean([x[0] for x in cover])),"2sigma":float(np.mean([x[1] for x in cover]))}

    # Conservative readiness gate: improvement over MA11, positive sample-level delta,
    # and uncertainty not catastrophically under-dispersed. Cross-target validation
    # remains a required future experiment.
    ready=(neural_advantage>0 and bootstrap["ci95"][0]>0 and coverage["1sigma"]>=0.55 and coverage["2sigma"]>=0.75)
    report={
        "stage":"9_final_scientific_validation",
        "model_checkpoint":str(model_path),
        "test_set_used_only_for_final_evaluation":True,
        "test_samples":len(ds),
        "aggregate_metrics":aggregate_metrics,
        "relative_rmse_improvement_vs_raw":raw_advantage,
        "relative_rmse_improvement_vs_ma11":neural_advantage,
        "sample_level_ma11_minus_neural_rmse_bootstrap":bootstrap,
        "uncertainty_coverage":coverage,
        "structure_metrics":shape_agg,
        "readiness_gate":{"passed":bool(ready),"criteria":{"beats_ma11":True,"positive_95pct_sample_advantage":True,"min_1sigma_coverage":0.55,"min_2sigma_coverage":0.75}},
        "scientific_status":"pre-integration candidate" if ready else "not ready for integration",
        "limitations":[
            "All synthetic clean targets derive from one WASP-39b reference candidate.",
            "This is not evidence of cross-planet generalization; an independent reference spectrum is still required.",
            "Uncertainty coverage is evaluated on synthetic held-out data and is not a substitute for observational calibration.",
            "No atmospheric feature or molecular detection is inferred by this benchmark.",
        ],
        "integration_policy":"Do not modify GUI or FastAPI recovery route until the external validation protocol is exercised with an independently sourced compatible reference spectrum.",
        "seed":seed,
    }
    Path(output).parent.mkdir(parents=True,exist_ok=True); Path(output).write_text(json.dumps(report,indent=2),encoding="utf-8")
    return report

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--model",default="artifacts/recovery_model/model.pt"); p.add_argument("--test",default="data/wasp39b/training/test.json"); p.add_argument("--output",default="artifacts/recovery_model/stage9_final_validation.json"); a=p.parse_args(); print(json.dumps(validate(a.model,a.test,a.output),indent=2))
