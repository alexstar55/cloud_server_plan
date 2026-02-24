from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt


def read_csv(path: str) -> dict[str, list[float]]:
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
    return cols


def has_col(data: dict[str, list[float]], k: str) -> bool:
    return k in data and len(data[k]) > 0


def _nan_minmax(x: list[float]) -> tuple[float, float] | None:
    vals = [v for v in x if not (v is None or math.isnan(v))]
    if not vals:
        return None
    return min(vals), max(vals)


def _set_tight_ylim(ax, y: list[float], pad_frac: float = 0.15, min_pad: float = 1e-6) -> None:
    mm = _nan_minmax(y)
    if mm is None:
        return
    lo, hi = mm
    if lo == hi:
        pad = max(min_pad, abs(lo) * 1e-6)
    else:
        pad = max(min_pad, (hi - lo) * pad_frac)
    ax.set_ylim(lo - pad, hi + pad)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--log_grad", action="store_true", help="Log-scale grad_norm axis (recommended).")
    ap.add_argument("--title", type=str, default="")
    ap.add_argument("--show_train_loss", action="store_true", help="Also plot train_loss (faint).")
    args = ap.parse_args()

    data = read_csv(args.csv)
    step = data["step"]

    fig, (ax_top, ax_bot) = plt.subplots(
        2,
        1,
        figsize=(10.5, 6.2),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.0]},
    )

    # ---------- Top panel: loss + grad_norm ----------
    if args.show_train_loss and has_col(data, "train_loss"):
        ax_top.plot(step, data["train_loss"], label="train_loss", linewidth=0.9, alpha=0.35, color="tab:blue")

    if has_col(data, "loss_ema"):
        ax_top.plot(step, data["loss_ema"], label="loss_ema", linewidth=1.2, color="tab:blue")
    ax_top.set_ylabel("loss (EMA)")

    ax_top_g = ax_top.twinx()
    if has_col(data, "grad_norm"):
        ax_top_g.plot(step, data["grad_norm"], label="grad_norm", linewidth=1.0, color="tab:green")
    ax_top_g.set_ylabel("grad_norm")
    if args.log_grad:
        ax_top_g.set_yscale("log")

    # ---------- Bottom panel: D_BM + d_pre_marg (separate y-axes) ----------
    # Prefer mass-normalized variant if available
    if has_col(data, "d_bm_q"):
        ax_bot.plot(step, data["d_bm_q"], label="D_BM_Q", linewidth=1.2, color="tab:red")
        ax_bot.set_ylabel("D_BM_Q")
        _set_tight_ylim(ax_bot, data["d_bm_q"])
        # Optionally also show raw D_BM faintly for comparison
        if has_col(data, "d_bm"):
            ax_bot.plot(step, data["d_bm"], label="D_BM (raw)", linewidth=1.0, alpha=0.25, color="tab:red")
    elif has_col(data, "d_bm"):
        ax_bot.plot(step, data["d_bm"], label="D_BM", linewidth=1.2, color="tab:red")
        ax_bot.set_ylabel("D_BM")
        _set_tight_ylim(ax_bot, data["d_bm"])

    ax_bot_r = ax_bot.twinx()
    if has_col(data, "d_pre_marg"):
        ax_bot_r.plot(step, data["d_pre_marg"], label="d_pre_marg", linewidth=1.2, color="tab:purple")
        ax_bot_r.set_ylabel("d_pre_marg")

    ax_bot.set_xlabel("step")

    if args.title:
        fig.suptitle(args.title)

    # Legends: keep clean (one per panel)
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
    print(f"Saved plot: {out_path}")


if __name__ == "__main__":
    main()