from __future__ import annotations

import argparse
import csv
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


def run(cmd: list[str], cwd: Path) -> None:
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(cwd))


def read_cols(path: Path) -> dict[str, np.ndarray]:
    cols: dict[str, list[float]] = {}
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            for k, v in row.items():
                cols.setdefault(k, [])
                if v is None or v == "":
                    cols[k].append(float("nan"))
                else:
                    cols[k].append(float(v))
    return {k: np.asarray(v, dtype=float) for k, v in cols.items()}


def nan_mean_std(x: np.ndarray) -> tuple[float, float]:
    return float(np.nanmean(x)), float(np.nanstd(x))


def compute_threshold(x: np.ndarray, baseline_idx: Iterable[int], k: float) -> float:
    xb = x[list(baseline_idx)]
    mu, sigma = nan_mean_std(xb)
    return mu + k * sigma


def first_spike_time(x: np.ndarray, theta: float, start_idx: int) -> int | None:
    for i in range(start_idx, len(x)):
        v = float(x[i])
        if (not math.isnan(v)) and v > theta:
            return i
    return None


def fmt_float(x: float) -> str:
    if math.isnan(x):
        return "nan"
    if abs(x) >= 1000 or (abs(x) > 0 and abs(x) < 1e-3):
        return f"{x:.3e}"
    return f"{x:.6f}"


def weight_tag(w: float) -> str:
    # stable folder name: w0, w0p01, w0p1, w1, ...
    s = f"{w:g}"
    s = s.replace("-", "m").replace(".", "p")
    return f"w{s}"


def latex_escape(s: str) -> str:
    return (
        s.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("&", "\\&")
        .replace("%", "\\%")
    )


def pretty_monitor_name(m: str) -> str:
    mapping = {
        "d_bm": r"$\mathcal{D}_{\text{BM}}(S_t)$",
        "d_bm_q": r"$\mathcal{D}_{\text{BM}}^{(Q)}(S_t)$",
        "d_pre_marg": r"$d_{\text{marg}}(Q_t)$",
        "grad_norm": r"Grad-norm",
        "loss_ema": r"Loss-EMA",
        "train_loss": r"Train-loss",
        "bm_reg": r"$\mathrm{bm\_reg}(S_t)$",
        "total_loss": r"Total-loss",
    }
    return mapping.get(m, latex_escape(m))


@dataclass
class AggResult:
    weight: float
    monitor: str
    hit_rate: float
    lead_mean: float
    lead_std: float


def aggregate_table2(csv_paths: list[Path], *, monitors: list[str], k: float, shock_step: int, fail_signal: str) -> list[AggResult]:
    if len(csv_paths) == 0:
        return []

    # per monitor
    agg_hits: dict[str, list[int]] = {m: [] for m in monitors}
    agg_leads: dict[str, list[float]] = {m: [] for m in monitors}

    for p in csv_paths:
        data = read_cols(p)

        if fail_signal not in data:
            raise SystemExit(f"fail_signal '{fail_signal}' not found in {p.name}; cols={list(data.keys())}")

        t0 = shock_step
        if t0 <= 1:
            raise SystemExit(f"shock_step too small: {t0}")

        baseline = range(0, t0)

        theta_fail = compute_threshold(data[fail_signal], baseline, k)
        t_fail = first_spike_time(data[fail_signal], theta_fail, start_idx=t0)
        if t_fail is None:
            t_fail = t0  # fallback same as table2_from_csv.py

        for m in monitors:
            if m not in data:
                agg_hits[m].append(0)
                continue
            theta = compute_threshold(data[m], baseline, k)
            t_event = first_spike_time(data[m], theta, start_idx=t0)
            hit = 1 if t_event is not None else 0
            agg_hits[m].append(hit)
            if hit:
                agg_leads[m].append(float(t_fail - t_event))

    n_runs = len(csv_paths)
    out: list[AggResult] = []
    for m in monitors:
        hit_rate = float(sum(agg_hits[m])) / float(n_runs)
        leads = np.asarray(agg_leads[m], dtype=float)
        if leads.size == 0:
            lead_mean = float("nan")
            lead_std = float("nan")
        else:
            lead_mean = float(np.mean(leads))
            lead_std = float(np.std(leads))
        out.append(AggResult(weight=float("nan"), monitor=m, hit_rate=hit_rate, lead_mean=lead_mean, lead_std=lead_std))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", type=str, default="runs/c_weight_sweep_norm_on")
    ap.add_argument("--weights", type=str, default="0,0.01,0.03,0.05,0.1",
                    help="comma-separated bm_reg_weight values (include 0 for baseline)")
    ap.add_argument("--n_seeds", type=int, default=10)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--normalize_features", action="store_true")
    ap.add_argument("--log_pre_marg", action="store_true")
    ap.add_argument("--bm_reg_variant", type=str, default="d_bm_q", choices=["d_bm", "d_bm_q"])

    # training knobs (same defaults as train_mve.py)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--shock_step", type=int, default=700)
    ap.add_argument("--shock_lr_mult", type=float, default=20.0)
    ap.add_argument("--tau", type=float, default=0.25)
    ap.add_argument("--sinkhorn_iters", type=int, default=30)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)

    # table2-style reporting
    ap.add_argument("--k", type=float, default=5.0)
    ap.add_argument("--fail_signal", type=str, default="loss_ema")
    ap.add_argument("--monitors", type=str, default="d_bm_q,grad_norm,d_pre_marg",
                    help="comma-separated monitor columns to aggregate")
    ap.add_argument("--skip_train", action="store_true")

    args = ap.parse_args()

    weights = [float(x.strip()) for x in args.weights.split(",") if x.strip() != ""]
    monitors = [m.strip() for m in args.monitors.split(",") if m.strip()]

    script_dir = Path(__file__).resolve().parent
    train_py = script_dir / "train_mve.py"

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    base_common = [
        sys.executable,
        str(train_py),
        "--device", args.device,
        "--steps", str(args.steps),
        "--shock_step", str(args.shock_step),
        "--shock_lr_mult", str(args.shock_lr_mult),
        "--tau", str(args.tau),
        "--sinkhorn_iters", str(args.sinkhorn_iters),
        "--batch", str(args.batch),
        "--lr", str(args.lr),
    ]
    if args.normalize_features:
        base_common.append("--normalize_features")
    if args.log_pre_marg:
        base_common.append("--log_pre_marg")

    # 1) train all weights
    if not args.skip_train:
        for w in weights:
            w_root = out_root / weight_tag(w)
            w_root.mkdir(parents=True, exist_ok=True)
            for seed in range(args.n_seeds):
                run(
                    base_common
                    + [
                        "--out", str(w_root / f"seed_{seed}"),
                        "--seed", str(seed),
                        "--bm_reg_weight", str(w),
                        "--bm_reg_variant", args.bm_reg_variant,
                    ],
                    cwd=script_dir,
                )

    # 2) aggregate -> long CSV + LaTeX rows
    summary_csv = out_root / "summary.csv"
    summary_tex = out_root / "summary.tex"

    all_rows: list[tuple[float, str, float, float, float]] = []

    for w in weights:
        w_root = out_root / weight_tag(w)
        csvs = [w_root / f"seed_{seed}" / "metrics.csv" for seed in range(args.n_seeds)]
        agg = aggregate_table2(
            csvs,
            monitors=monitors,
            k=args.k,
            shock_step=args.shock_step,
            fail_signal=args.fail_signal,
        )
        for r in agg:
            all_rows.append((w, r.monitor, r.hit_rate, r.lead_mean, r.lead_std))

    with open(summary_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bm_reg_weight", "monitor", "hit_rate", "lead_mean", "lead_std", "k", "fail_signal", "shock_step", "n_seeds"])
        for (bw, mon, hr, lm, ls) in all_rows:
            w.writerow([bw, mon, hr, lm, ls, args.k, args.fail_signal, args.shock_step, args.n_seeds])

    with open(summary_tex, "w", newline="") as f:
        f.write("% Paste rows into a LaTeX tabular.\n")
        f.write("% Columns: weight & monitor & hit rate & lead time (mean±std) \\\\\n")
        for (bw, mon, hr, lm, ls) in all_rows:
            monitor_tex = pretty_monitor_name(mon)
            hit_tex = f"{hr:.2f}"
            if math.isnan(lm) or math.isnan(ls):
                lead_tex = "N/A"
            else:
                lead_tex = rf"${lm:.1f}\pm{ls:.1f}$"
            f.write(rf"{bw:g} & {monitor_tex} & {hit_tex} & {lead_tex} \\" + "\n")

    print(f"\nWrote:\n- {summary_csv}\n- {summary_tex}")


if __name__ == "__main__":
    main()