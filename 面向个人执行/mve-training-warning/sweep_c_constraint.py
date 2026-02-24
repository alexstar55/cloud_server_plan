from __future__ import annotations

import argparse
import csv
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

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


def _nanmean(x: np.ndarray) -> float:
    return float(np.nanmean(x))


def _nanstd(x: np.ndarray) -> float:
    return float(np.nanstd(x))


def _nanmax(x: np.ndarray) -> float:
    return float(np.nanmax(x))


@dataclass
class SignalSummary:
    name: str
    baseline_mean_mu: float
    baseline_mean_sd: float
    baseline_std_mu: float
    baseline_std_sd: float
    post_max_mu: float
    post_max_sd: float
    max_over_base_mu: float
    max_over_base_sd: float


def summarize_group(csv_paths: list[Path], t0: int, signals: list[str]) -> list[SignalSummary]:
    eps = 1e-12
    out: list[SignalSummary] = []

    for s in signals:
        base_means: list[float] = []
        base_stds: list[float] = []
        post_maxs: list[float] = []
        ratios: list[float] = []

        for p in csv_paths:
            d = read_cols(p)
            if s not in d:
                continue
            x = d[s]
            xb = x[:t0]
            xa = x[t0:]

            bm = _nanmean(xb)
            bs = _nanstd(xb)
            pm = _nanmax(xa)

            base_means.append(bm)
            base_stds.append(bs)
            post_maxs.append(pm)
            ratios.append(pm / (bm + eps))

        if len(base_means) == 0:
            continue

        out.append(
            SignalSummary(
                name=s,
                baseline_mean_mu=float(np.mean(base_means)),
                baseline_mean_sd=float(np.std(base_means)),
                baseline_std_mu=float(np.mean(base_stds)),
                baseline_std_sd=float(np.std(base_stds)),
                post_max_mu=float(np.mean(post_maxs)),
                post_max_sd=float(np.std(post_maxs)),
                max_over_base_mu=float(np.mean(ratios)),
                max_over_base_sd=float(np.std(ratios)),
            )
        )

    return out


def print_group_report(title: str, csv_paths: list[Path], t0: int, signals: list[str]) -> None:
    print("\n" + "=" * 88)
    print(title)
    print(f"runs: {len(csv_paths)}    shock_step: {t0}")
    print("-" * 88)
    summaries = summarize_group(csv_paths, t0=t0, signals=signals)
    if not summaries:
        print("No signals found.")
        return

    header = (
        f"{'signal':<12s} | "
        f"{'base_mean (μ±σ)':<22s} | "
        f"{'base_std (μ±σ)':<22s} | "
        f"{'post_max (μ±σ)':<22s} | "
        f"{'max/base (μ±σ)':<18s}"
    )
    print(header)
    print("-" * len(header))

    for s in summaries:
        print(
            f"{s.name:<12s} | "
            f"{s.baseline_mean_mu:9.4g}±{s.baseline_mean_sd:8.2g} | "
            f"{s.baseline_std_mu:9.4g}±{s.baseline_std_sd:8.2g} | "
            f"{s.post_max_mu:9.4g}±{s.post_max_sd:8.2g} | "
            f"{s.max_over_base_mu:7.3g}±{s.max_over_base_sd:6.2g}"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", type=str, default="runs/c_constraint_norm_on")
    ap.add_argument("--n_seeds", type=int, default=10)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--normalize_features", action="store_true")
    ap.add_argument("--log_pre_marg", action="store_true")

    ap.add_argument("--bm_reg_weight_on", type=float, default=0.10)
    ap.add_argument("--bm_reg_variant", type=str, default="d_bm_q", choices=["d_bm", "d_bm_q"])

    # keep same MVE defaults unless overridden
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--shock_step", type=int, default=700)
    ap.add_argument("--shock_lr_mult", type=float, default=20.0)
    ap.add_argument("--tau", type=float, default=0.25)
    ap.add_argument("--sinkhorn_iters", type=int, default=30)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)

    # reporting knobs
    ap.add_argument("--k", type=float, default=5.0, help="threshold = mu + k*sigma (baseline window)")
    ap.add_argument(
        "--monitors",
        type=str,
        default="d_bm_q,grad_norm,d_pre_marg,bm_reg",
        help="comma-separated monitors passed to table2_from_csv.py",
    )
    ap.add_argument(
        "--signals",
        type=str,
        default="loss_ema,train_loss,grad_norm,d_bm_q,bm_reg,total_loss",
        help="comma-separated signals for baseline/post_max/max-over-base report",
    )
    ap.add_argument("--skip_train", action="store_true")
    ap.add_argument("--skip_plots", action="store_true")
    ap.add_argument("--skip_table2", action="store_true")
    ap.add_argument("--print_tex", action="store_true", help="ask table2_from_csv.py to print LaTeX rows")
    ap.add_argument("--print_md", action="store_true", help="ask table2_from_csv.py to print Markdown rows")
    ap.add_argument("--fail_signal", type=str, default="loss_ema")

    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent

    out_root = Path(args.out_root)
    off_root = out_root / "constraint_off"
    on_root = out_root / "constraint_on"
    (off_root / "agg").mkdir(parents=True, exist_ok=True)
    (on_root / "agg").mkdir(parents=True, exist_ok=True)

    train_py = script_dir / "train_mve.py"
    plot_py = script_dir / "plot_metrics_agg.py"
    table_py = script_dir / "table2_from_csv.py"

    base_common = [
        sys.executable,
        str(train_py),
        "--device",
        args.device,
        "--steps",
        str(args.steps),
        "--shock_step",
        str(args.shock_step),
        "--shock_lr_mult",
        str(args.shock_lr_mult),
        "--tau",
        str(args.tau),
        "--sinkhorn_iters",
        str(args.sinkhorn_iters),
        "--batch",
        str(args.batch),
        "--lr",
        str(args.lr),
    ]
    if args.normalize_features:
        base_common.append("--normalize_features")
    if args.log_pre_marg:
        base_common.append("--log_pre_marg")

    if not args.skip_train:
        for seed in range(args.n_seeds):
            run(
                base_common
                + [
                    "--out",
                    str(off_root / f"seed_{seed}"),
                    "--seed",
                    str(seed),
                    "--bm_reg_weight",
                    "0.0",
                    "--bm_reg_variant",
                    args.bm_reg_variant,
                ],
                cwd=script_dir,
            )
            run(
                base_common
                + [
                    "--out",
                    str(on_root / f"seed_{seed}"),
                    "--seed",
                    str(seed),
                    "--bm_reg_weight",
                    str(args.bm_reg_weight_on),
                    "--bm_reg_variant",
                    args.bm_reg_variant,
                ],
                cwd=script_dir,
            )

    off_csvs = [off_root / f"seed_{seed}" / "metrics.csv" for seed in range(args.n_seeds)]
    on_csvs = [on_root / f"seed_{seed}" / "metrics.csv" for seed in range(args.n_seeds)]

    if not args.skip_plots:
        run(
            [
                sys.executable,
                str(plot_py),
                "--csv",
                *[str(p) for p in off_csvs],
                "--out",
                str(off_root / "agg" / "plot_agg.png"),
                "--title",
                f"C (constraint OFF): N={args.n_seeds}",
                "--log_grad",
                "--show_raw_dbm",
                "--shock_step",
                str(args.shock_step),
            ],
            cwd=script_dir,
        )
        run(
            [
                sys.executable,
                str(plot_py),
                "--csv",
                *[str(p) for p in on_csvs],
                "--out",
                str(on_root / "agg" / "plot_agg.png"),
                "--title",
                f"C (constraint ON, w={args.bm_reg_weight_on}): N={args.n_seeds}",
                "--log_grad",
                "--show_raw_dbm",
                "--shock_step",
                str(args.shock_step),
            ],
            cwd=script_dir,
        )

    # numeric report (avoid "describe the picture")
    signals = [s.strip() for s in args.signals.split(",") if s.strip()]
    print_group_report("C summary: constraint OFF", off_csvs, t0=args.shock_step, signals=signals)
    print_group_report("C summary: constraint ON", on_csvs, t0=args.shock_step, signals=signals)

    # table2 style report (hit-rate / lead-time)
    if not args.skip_table2:
        monitors = args.monitors
        extra = []
        if args.print_tex:
            extra.append("--print_tex")
        if args.print_md:
            extra.append("--print_md")

        print("\n" + "=" * 88)
        print("Table2-style summary: constraint OFF")
        run(
            [
                sys.executable,
                str(table_py),
                "--csv",
                *[str(p) for p in off_csvs],
                "--k",
                str(args.k),
                "--shock_step",
                str(args.shock_step),
                "--fail_signal",
                args.fail_signal,
                "--monitors",
                monitors,
                *extra,
            ],
            cwd=script_dir,
        )

        print("\n" + "=" * 88)
        print("Table2-style summary: constraint ON")
        run(
            [
                sys.executable,
                str(table_py),
                "--csv",
                *[str(p) for p in on_csvs],
                "--k",
                str(args.k),
                "--shock_step",
                str(args.shock_step),
                "--fail_signal",
                args.fail_signal,
                "--monitors",
                monitors,
                *extra,
            ],
            cwd=script_dir,
        )

    print(
        f"\nDone. Figures:\n- {off_root / 'agg' / 'plot_agg.png'}\n- {on_root / 'agg' / 'plot_agg.png'}"
    )


if __name__ == "__main__":
    main()