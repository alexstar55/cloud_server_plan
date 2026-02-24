from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import matplotlib.pyplot as plt


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
    return {k: np.asarray(v, dtype=float) for k, v in cols.items()}


def infer_shock_step_from_lr(lr: np.ndarray) -> int | None:
    if lr.size < 2:
        return None
    lr0 = lr[0]
    for i in range(1, lr.size):
        if (not math.isnan(lr[i])) and lr[i] != lr0:
            return i
    return None


def stack_col(datas: list[dict[str, np.ndarray]], key: str) -> np.ndarray | None:
    arrs = []
    for d in datas:
        if key not in d:
            return None
        arrs.append(d[key])
    # all same length check
    n = arrs[0].shape[0]
    if any(a.shape[0] != n for a in arrs):
        raise ValueError(f"length mismatch for column '{key}' across CSVs")
    return np.stack(arrs, axis=0)  # (R, T)


def nan_mean_std(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = np.nanmean(x, axis=0)
    sd = np.nanstd(x, axis=0)
    return mu, sd


def plot_band(ax, step, mu, sd, label, color, alpha=0.18, lw=1.2):
    ax.plot(step, mu, label=label, color=color, linewidth=lw)
    ax.fill_between(step, mu - sd, mu + sd, color=color, alpha=alpha, linewidth=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, nargs="+", required=True, help="One or more metrics.csv paths (multiple seeds).")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--title", type=str, default="")
    ap.add_argument("--log_grad", action="store_true")
    ap.add_argument("--shock_step", type=int, default=-1, help="If <0, infer from lr jump in first CSV.")
    ap.add_argument("--show_raw_dbm", action="store_true", help="Also plot D_BM (raw) faintly if available.")
    args = ap.parse_args()

    datas = [read_cols(p) for p in args.csv]
    if "step" not in datas[0]:
        raise SystemExit("missing 'step' column")
    step = datas[0]["step"]

    # infer shock step
    if args.shock_step >= 0:
        t0 = args.shock_step
    else:
        lr0 = datas[0].get("lr")
        t0 = infer_shock_step_from_lr(lr0) if lr0 is not None else None

    loss = stack_col(datas, "loss_ema")
    grad = stack_col(datas, "grad_norm")
    dbmq = stack_col(datas, "d_bm_q")
    dbm = stack_col(datas, "d_bm")  # optional
    dpm = stack_col(datas, "d_pre_marg")  # optional

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(10.5, 6.2), sharex=True, gridspec_kw={"height_ratios": [1.0, 1.0]}
    )

    # ---- top: loss + grad ----
    if loss is not None:
        mu, sd = nan_mean_std(loss)
        plot_band(ax_top, step, mu, sd, "loss_ema (mean±std)", "tab:blue")
        ax_top.set_ylabel("loss (EMA)")

    ax_top_g = ax_top.twinx()
    if grad is not None:
        mu, sd = nan_mean_std(grad)
        plot_band(ax_top_g, step, mu, sd, "grad_norm (mean±std)", "tab:green")
        ax_top_g.set_ylabel("grad_norm")
        if args.log_grad:
            ax_top_g.set_yscale("log")

    # ---- bottom: D_BM_Q + optional D_BM + d_pre_marg ----
    if dbmq is not None:
        mu, sd = nan_mean_std(dbmq)
        plot_band(ax_bot, step, mu, sd, "D_BM_Q (mean±std)", "tab:red")
        ax_bot.set_ylabel("D_BM_Q")

    if args.show_raw_dbm and (dbm is not None):
        mu, sd = nan_mean_std(dbm)
        ax_bot.plot(step, mu, label="D_BM raw (mean)", color="tab:red", alpha=0.30, linewidth=1.0)

    ax_bot_r = ax_bot.twinx()
    if dpm is not None:
        mu, sd = nan_mean_std(dpm)
        plot_band(ax_bot_r, step, mu, sd, "d_pre_marg (mean±std)", "tab:purple")
        ax_bot_r.set_ylabel("d_pre_marg")

    if t0 is not None:
        for ax in (ax_top, ax_bot):
            ax.axvline(float(t0), color="k", linestyle="--", linewidth=1.0, alpha=0.45)

    ax_bot.set_xlabel("step")
    if args.title:
        fig.suptitle(args.title)

    # legends
    l1, lab1 = ax_top.get_legend_handles_labels()
    l2, lab2 = ax_top_g.get_legend_handles_labels()
    if l1 or l2:
        ax_top.legend(l1 + l2, lab1 + lab2, loc="upper right")

    l3, lab3 = ax_bot.get_legend_handles_labels()
    l4, lab4 = ax_bot_r.get_legend_handles_labels()
    if l3 or l4:
        ax_bot.legend(l3 + l4, lab3 + lab4, loc="upper right")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f"Saved aggregate plot: {out_path}")


if __name__ == "__main__":
    main()