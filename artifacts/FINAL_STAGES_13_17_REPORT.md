# OurionSpectra — Stages 13–17 Final Report

## Executive decision

The project has been taken through multi-target data preparation, model retraining, independent validation, integration hardening, and final regression testing.

**The experimental multi-target neural model is NOT approved as the default recovery engine.** The transparent existing recovery baseline remains the production/default path. The neural model is available only through an explicit environment-variable opt-in.

This gate is intentional: the neural model does not consistently outperform the classical baseline across the multi-target benchmark and independent HAT-P-1b evaluation.

## Stage 13 — Multi-target corpus

Reference candidates used for training:
- WASP-39b: existing OurionSpectra reference candidate
- HAT-P-26b: Wakeford et al. 2017 published transmission-spectrum table
- WASP-17b: Alderson et al. 2022 published transmission-spectrum table

External holdout:
- HAT-P-1b: Wakeford et al. 2013, never used for training/validation/model selection

Synthetic observations use reported per-channel uncertainties, independent Gaussian measurement noise, and a controlled correlated component scaled by the empirical uncertainty.

Total generated realizations: 2,700
- Train: 1,800
- Validation: 450
- Test: 450
- 600/150/150 realizations per training target

No atmospheric features were added. Published observations remain reference candidates, not ground truth.

## Stage 14 — Multi-target model

A lightweight spectral recovery MLP was trained on balanced point contributions from the three training targets. Inputs include wavelength, noisy flux, empirical uncertainty, local baseline, residual-to-baseline, gradient and curvature features, and a mask/bias channel.

Training points: 180,000 balanced sampled points
Validation points: 180,000 balanced sampled points
Epochs: 15

The model checkpoint is experimental only.

## Stage 15 — Scientific validation

The model was evaluated at multiple synthetic noise scales on all three training-domain targets and on the untouched HAT-P-1b target.

Mean RMSE across noise scales:

| Target | Raw | Classical baseline | Neural |
|---|---:|---:|---:|
| WASP-39b | 0.01290 | 0.00701 | 0.00799 |
| HAT-P-26b | 0.000136 | 0.000114 | 0.000501 |
| WASP-17b | 0.000610 | 0.000361 | 0.000406 |
| HAT-P-1b external | 0.000242 | 0.000173 | 0.000241 |

The neural candidate therefore does not meet the criterion of consistently beating the transparent baseline.

Mean 1-sigma coverage:
- WASP-39b: ~77%
- HAT-P-26b: ~5.5%
- WASP-17b: ~65%
- HAT-P-1b external: ~65%

Uncertainty calibration is therefore not yet scientifically acceptable across targets.

## Stage 16 — Safe integration

The existing FastAPI architecture and GUI layout were not changed.

`ourionspectra/model.py` now has a safe opt-in seam for the experimental multi-target model through:

`OURIONSPECTRA_ENABLE_EXPERIMENTAL_ML=1`

Default behavior remains unchanged. This prevents an unvalidated model from silently becoming the application default.

No atmospheric feature detector was enabled; the project continues to return an honest empty feature state until a defensible detector exists.

## Stage 17 — Final regression/documentation

Complete test suite:

**82 / 82 tests passed.**

The suite includes science/math, FastAPI, training-data, recovery-model, domain-robustness, external-validation, multi-target corpus, and experimental integration tests.

## Scientific limitations

1. The training corpus remains small by modern astronomical ML standards.
2. The three training targets combine different instruments, reductions and wavelength samplings; this is useful domain diversity but not a homogeneous instrument model.
3. The synthetic noise process is not a detector-level JWST time-series simulator.
4. Published transmission spectra are observational reference candidates and are not analytical ground truth.
5. HAT-P-1b external evaluation is a synthetic-noise stress test around a published spectrum, not recovery from the original raw time-series exposures.
6. The neural model currently does not demonstrate consistent superiority over a transparent smoothing baseline.
7. Molecular/atmospheric feature detection remains disabled; no feature claims are fabricated.

## Selection-ready scientific claim

A defensible project statement at this stage is:

> OurionSpectra is an uncertainty-aware, noise-recovery framework for exoplanet transmission spectra. It combines empirically anchored synthetic observations, multi-target reference candidates, classical baselines, an experimental neural recovery model, held-out evaluation, and independent-target validation. The neural component remains experimental because cross-target performance and uncertainty calibration have not yet justified replacing the transparent baseline.
