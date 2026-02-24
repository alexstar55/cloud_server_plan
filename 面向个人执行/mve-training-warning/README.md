# MVE: Training Warning via Sinkhorn Deviation

Minimal Viable Experiment (MVE) for the `D_BM` training-instability monitoring signal described in `The Diagnostic Turn: Detecting AI Anomalies and Deepfakes via Mathematical Invariant Violations`.

## What it does
- Trains a tiny MLP on synthetic nonlinear regression
- Builds a batch-level score matrix `S_t` from penultimate features (a self-transport / similarity operator)
- Computes Sinkhorn projection to prescribed marginals and logs the deviation metric `D_BM(S_t)`
- Logs baseline monitors: loss, loss EMA, gradient norm
- Saves a CSV and a quick plot

## Requirements
- Python 3.10+
- Packages: `torch`, `numpy`, `matplotlib`

Install:
```bash
pip install -r requirements.txt
```

## Run
```bash
# Feature normalization ON
# "cd mve-training-warning" first, for all .py scripts run below
python3 train_mve.py --out runs/norm_on/seed_0 --seed 0 --log_pre_marg --normalize_features
python3 plot_metrics.py --csv runs/norm_on/seed_0/metrics.csv --out runs/norm_on/seed_0/plot.png --log_grad

# Feature normalization OFF
python3 train_mve.py --out runs/norm_off/seed_0 --seed 0 --log_pre_marg
python3 plot_metrics.py --csv runs/norm_off/seed_0/metrics.csv --out runs/norm_off/seed_0/plot.png --log_grad
```

## C: Constraint + Diagnostic (closed-loop)

This experiment uses the same deviation signal both as:
- a training-time regularizer (`bm_reg_weight * bm_reg`), and
- a monitoring signal logged to CSV.

Run (N=10 seeds, feature normalization ON):
```bash
python3 sweep_c_constraint.py  \
  --out_root runs/c_norm_on \
  --n_seeds 10 \
  --normalize_features \
  --log_pre_marg \
  --bm_reg_weight_on 0.1 \
  --print_tex
```

## C: Regularization weight sweep (writes summary.csv / summary.tex)

This experiment sweeps bm_reg_weight and aggregates hit-rate / lead-time across seeds.

Run (N=10 seeds, feature normalization ON):
```bash
python3 sweep_c_weights.py \
--out_root runs/c_weight_sweep_norm_on \
--weights 0,0.01,0.03,0.05,0.1 \
--n_seeds 10 \
--device cpu \
--normalize_features \
--log_pre_marg \
--bm_reg_variant d_bm_q
```
Outputs:
- runs/c_weight_sweep_norm_on/summary.csv
- runs/c_weight_sweep_norm_on/summary.tex

## Notes
- This is intentionally small and self-contained.
- If you already have a heavy environment (e.g., `mmdet3d`) with PyTorch + NumPy + Matplotlib installed, you can run this in that environment; a clean venv is usually less fragile.
