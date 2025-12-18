# Robcontrol Pipeline

This submodule provides a small end-to-end pipeline for conformalized robust LQR experiments.

## 1) Generate a dataset
Synthetic tasks: `mass_spring_damper`, `cartpole`, `dc_motor`.
```bash
python robcontrol/data.py \
  --task cartpole \
  --num-samples 500 \
  --horizon 200 \
  --seed 0 \
  --out robcontrol/artifacts/cartpole_dataset.npz
```
The NPZ stores `thetas`, `A_true`, `B_true`, `A_hat`, `B_hat`, and cost matrices `q`, `r`.

## 2) Train a dynamics predictor
Trains the improved MLP on the saved dataset and writes weights + metadata.
```bash
python robcontrol/model.py \
  --dataset robcontrol/artifacts/cartpole_dataset.npz \
  --out-model robcontrol/artifacts/cartpole_model.pt \
  --out-meta robcontrol/artifacts/cartpole_meta.json \
  --epochs 400 \
  --lr 1e-3 \
  --batch-size 128
```

## 3) Calibrate and assess nominal vs CPC
Loads dataset + model, calibrates with the operator-norm score, plots calibration, and evaluates nominal vs CPC (both synthesized on predicted dynamics).
```bash
python robcontrol/assess.py \
  --dataset robcontrol/artifacts/cartpole_dataset.npz \
  --model robcontrol/artifacts/cartpole_model.pt \
  --meta robcontrol/artifacts/cartpole_meta.json \
  --horizon 200 \
  --trials 5 \
  --rollouts 5 \
  --seed 0
```
Outputs:
- Calibration plot: `<dataset>_calibration.png`
- Metrics JSON: `<dataset>_metrics.json` (means/stds; paired t-test robust< nominal when trials > 1).

## Sample Results

Cartpole (500 samples, horizon 200, rollouts 5, trials 5):

| method                | mean cost | std    |
|-----------------------|-----------|--------|
| true_opt_on_true      | 132.28    | 9.10   |
| nominal_on_true       | 3353.10   | 456.71 |
| cpc_on_true           | 3353.08   | 456.70 |
| t-test (robust < nom) | t = -4.18 | p=0.007 |

## Notes
- Scripts prepend the repo root to `PYTHONPATH`; run them from the repo root.
- CPC controller lives in `robcontrol/controllers/cpc.py`. Utilities for LQR/Riccati/rollouts are in `robcontrol/utils.py`.
