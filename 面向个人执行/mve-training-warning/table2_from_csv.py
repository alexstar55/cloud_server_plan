from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass
class SpikeResult:
    name: str
    theta: float
    t_event: int | None  # None means "no spike"
    hit: int


def read_cols(path: str) -> dict[str, np.ndarray]:
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
    return {k: np.array(v, dtype=float) for k, v in cols.items()}


def nan_mean_std(x: np.ndarray) -> tuple[float, float]:
    mu = float(np.nanmean(x))
    sigma = float(np.nanstd(x))
    return mu, sigma


def first_spike_time(x: np.ndarray, theta: float, start_idx: int) -> int | None:
    for i in range(start_idx, len(x)):
        v = x[i]
        if not math.isnan(v) and v > theta:
            return i
    return None


def infer_shock_step_from_lr(lr: np.ndarray) -> int | None:
    if len(lr) < 2:
        return None
    lr0 = lr[0]
    for i in range(1, len(lr)):
        if (not math.isnan(lr[i])) and lr[i] != lr0:
            return i
    return None


def compute_threshold(x: np.ndarray, baseline_idx: Iterable[int], k: float) -> float:
    xb = x[list(baseline_idx)]
    mu, sigma = nan_mean_std(xb)
    return mu + k * sigma


def fmt_float(x: float) -> str:
    if math.isnan(x):
        return "nan"
    if abs(x) >= 1000 or (abs(x) > 0 and abs(x) < 1e-3):
        return f"{x:.3e}"
    return f"{x:.6f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, nargs="+", required=True, help="One or more CSV files (multiple seeds/runs).")
    ap.add_argument("--k", type=float, default=5.0, help="Threshold = mean + k*std on baseline window.")
    ap.add_argument("--shock_step", type=int, default=-1, help="If <0, infer from lr jump.")
    ap.add_argument("--fail_signal", type=str, default="loss_ema", help="Which signal defines failure time.")
    ap.add_argument("--monitors", type=str, default="d_bm_q,grad_norm,d_pre_marg",
                    help="Comma-separated monitor names to evaluate.")
    ap.add_argument("--print_md", action="store_true", help="Print a Markdown table row block.")
    ap.add_argument("--print_tex", action="store_true", help="Print LaTeX tabular rows (one per monitor).")    
    ap.add_argument("--verbose", action="store_true", help="Print per-run debug details.")
    args = ap.parse_args()

    csv_paths = args.csv
    if len(csv_paths) == 0:
        raise SystemExit("No CSVs provided")

    monitors = [m.strip() for m in args.monitors.split(",") if m.strip()]
    # Aggregate: per monitor, collect leads (only when hit) and hits (0/1 for each run)
    agg_hits: dict[str, list[int]] = {m: [] for m in monitors}
    agg_leads: dict[str, list[float]] = {m: [] for m in monitors}

    # We also track per-run failure definition for debug
    for csv_path in csv_paths:
        data = read_cols(csv_path)
        step = data.get("step")
        if step is None:
            raise SystemExit(f"CSV missing 'step' column: {csv_path}")

        if args.shock_step >= 0:
            t0 = args.shock_step
        else:
            lr = data.get("lr")
            if lr is None:
                raise SystemExit("Need --shock_step or an 'lr' column to infer it.")
            t0 = infer_shock_step_from_lr(lr)
            if t0 is None:
                raise SystemExit(f"Could not infer shock_step from lr in: {csv_path}")

        # Failure time from chosen fail_signal
        if t0 <= 1:
            raise SystemExit(f"shock_step too small: {t0} ({csv_path})")

        baseline = range(0, t0)  # 0..t0-1

        # Failure time from chosen fail_signal
        fail_name = args.fail_signal
        if fail_name not in data:
            raise SystemExit(f"fail_signal '{fail_name}' not in CSV columns: {list(data.keys())}")

        theta_fail = compute_threshold(data[fail_name], baseline, args.k)
        t_fail = first_spike_time(data[fail_name], theta_fail, start_idx=t0)
        if t_fail is None:
            # Fallback: allow defining failure as first post-shock step if no spike
            t_fail = t0

        if args.verbose:
            print(f"csv: {csv_path}")
            print(f"shock_step t0: {t0}")
            print(f"failure signal: {fail_name}")
            print(f"theta_fail: {fmt_float(theta_fail)}")
            print(f"t_fail: {t_fail}")

        for m in monitors:
            if m not in data:
                agg_hits[m].append(0)
                continue
            theta = compute_threshold(data[m], baseline, args.k)
            t_event = first_spike_time(data[m], theta, start_idx=t0)
            hit = 1 if t_event is not None else 0
            agg_hits[m].append(hit)
            if hit:
                agg_leads[m].append(float(t_fail - t_event))
            if args.verbose:
                lead = float("nan") if t_event is None else float(t_fail - t_event)
                print(f"{m:>12s}  theta={fmt_float(theta)}  t_event={t_event}  lead={fmt_float(lead)}  hit={hit}")

    n_runs = len(csv_paths)
    print(f"runs: {n_runs}")
    print(f"fail_signal: {args.fail_signal}")
    print(f"k: {args.k}")
    print("---- aggregated ----")
    for m in monitors:
        hit_rate = float(sum(agg_hits[m])) / float(n_runs)
        leads = np.array(agg_leads[m], dtype=float)
        if leads.size == 0:
            lead_mean = float("nan")
            lead_std = float("nan")
        else:
            lead_mean = float(np.mean(leads))
            lead_std = float(np.std(leads))
        print(
            f"{m:>12s}  hit_rate={hit_rate:.3f}  "
            f"lead_mean={fmt_float(lead_mean)}  lead_std={fmt_float(lead_std)}"
        )

    # Print a Markdown block that you can paste into Table 2
    def latex_escape(s: str) -> str:
        # minimal escaping for LaTeX tabular
        return (
            s.replace("\\", "\\textbackslash{}")
            .replace("_", "\\_")
            .replace("&", "\\&")
            .replace("%", "\\%")
        )

    def pretty_monitor_name(m: str) -> str:
        # Map CSV column names to paper-friendly LaTeX labels
        mapping = {
            "d_bm": r"$\mathcal{D}_{\text{BM}}(S_t)$",
            "d_bm_q": r"$\mathcal{D}_{\text{BM}}^{(Q)}(S_t)$",
            "d_pre_marg": r"$d_{\text{marg}}(Q_t)$",
            "grad_norm": r"Grad-norm",
            "loss_ema": r"Loss-EMA",
            "train_loss": r"Train-loss",
        }
        return mapping.get(m, latex_escape(m))

    def pretty_note(m: str) -> str:
        mapping = {
            "d_bm": r"Sinkhorn projection distance",
            "d_bm_q": r"Sinkhorn distance on globally-normalized $Q_t$",
            "d_pre_marg": r"Pre-projection marginal $\ell_1$ deviation on $Q_t$",
            "grad_norm": r"$\|\nabla_\theta \ell_t\|_2$",
            "loss_ema": r"Failure indicator (reference)",
        }
        return mapping.get(m, "")

    # Print a Markdown block that you can paste into Table 2
    if args.print_md:
        print("\nMarkdown rows (paste into Table 2):")
        for m in monitors:
            hit_rate = float(sum(agg_hits[m])) / float(n_runs)
            leads = np.array(agg_leads[m], dtype=float)
            if leads.size == 0:
                lead_str = "N/A"
            else:
                lead_mean = float(np.mean(leads))
                lead_std = float(np.std(leads))
                lead_str = f"{lead_mean:.1f}±{lead_std:.1f}"
            theta_str = f"$\\mu+{args.k}\\sigma$"
            print(f"| {m} | {theta_str} | {hit_rate:.2f} | {lead_str} | {m} |")
 
    # Print LaTeX tabular rows (paste into Table 2 tabular)
    if args.print_tex:
        print("\nLaTeX tabular rows (paste into Table 2):")
        for m in monitors:
            monitor_tex = pretty_monitor_name(m)
            theta_tex = rf"$\mu+{args.k}\sigma$"

            hit_rate = float(sum(agg_hits[m])) / float(n_runs)
            hit_tex = f"{hit_rate:.2f}"

            leads = np.array(agg_leads[m], dtype=float)
            if leads.size == 0:
                lead_tex = "N/A"
            else:
                lead_mean = float(np.mean(leads))
                lead_std = float(np.std(leads))
                lead_tex = rf"${lead_mean:.1f}\pm{lead_std:.1f}$"

            note_tex = pretty_note(m)

            if note_tex == "":
                note_tex = latex_escape(m)


            # 5 columns:
            # Monitor & Spike threshold & Hit rate & Lead time (mean±std) & Notes \\
            print(f"{monitor_tex} & {theta_tex} & {hit_tex} & {lead_tex} & {note_tex} \\\\")
if __name__ == "__main__":
    main()