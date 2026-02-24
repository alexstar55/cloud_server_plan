from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from sinkhorn import d_bm_from_scores


class TinyMLP(nn.Module):
    def __init__(self, d_in: int = 8, d_hidden: int = 128, d_out: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.Tanh(),
            nn.Linear(d_hidden, d_hidden),
            nn.Tanh(),
        )
        self.head = nn.Linear(d_hidden, d_out)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x)
        y = self.head(h)
        return y, h


def make_dataset(n: int, d_in: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, d_in)).astype(np.float32)

    y = (
        np.sin(x[:, 0])
        + 0.5 * np.cos(1.7 * x[:, 1])
        + 0.2 * x[:, 2] * x[:, 3]
        + 0.1 * (x[:, 4] ** 2)
    ).astype(np.float32)
    y = y.reshape(-1, 1)
    return x, y


@dataclass
class EMA:
    beta: float
    value: float | None = None

    def update(self, x: float) -> float:
        if self.value is None:
            self.value = x
        else:
            self.value = self.beta * self.value + (1.0 - self.beta) * x
        return float(self.value)


def grad_norm(model: nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is None:
            continue
        total += float(torch.sum(p.grad.detach() ** 2).cpu())
    return math.sqrt(total)


def build_score_matrix(h: torch.Tensor, normalize_features: bool) -> torch.Tensor:
    # h: (B, d)
    if normalize_features:
        h_use = h / (h.norm(dim=1, keepdim=True) + 1e-12)
    else:
        h_use = h

    S = (h_use @ h_use.t()) / math.sqrt(h_use.shape[1])

    # mask diagonal out-of-place (autograd-safe)
    eye = torch.eye(S.shape[0], device=S.device, dtype=torch.bool)
    S_min = torch.amin(S)
    S = S.masked_fill(eye, S_min)
    return S


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="runs/run1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    ap.add_argument("--d_in", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--steps", type=int, default=1200)

    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--shock_step", type=int, default=700)
    ap.add_argument("--shock_lr_mult", type=float, default=20.0)

    ap.add_argument("--tau", type=float, default=0.25)
    ap.add_argument("--sinkhorn_iters", type=int, default=30)

    ap.add_argument(
        "--normalize_features",
        action="store_true",
        help="If set, L2-normalize batch features before building S_t.",
    )
    ap.add_argument(
        "--log_pre_marg",
        action="store_true",
        help="If set, also log a pre-projection marginal deviation of K_t (no Sinkhorn).",
    )

    # --- C方案：同一个 deviation 既做正则又做监控 ---
    ap.add_argument(
        "--bm_reg_weight",
        type=float,
        default=0.0,
        help="If >0, add bm_reg_weight * bm_reg(S_t) to training loss (constraint+diagnostic closed-loop).",
    )
    ap.add_argument(
        "--bm_reg_variant",
        type=str,
        default="d_bm_q",
        choices=["d_bm", "d_bm_q"],
        help="Which deviation to use as training regularizer.",
    )

    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    x_train, y_train = make_dataset(n=4096, d_in=args.d_in, seed=args.seed)
    x_val, y_val = make_dataset(n=1024, d_in=args.d_in, seed=args.seed + 1)

    device = torch.device(args.device)
    model = TinyMLP(d_in=args.d_in, d_hidden=args.hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    ema = EMA(beta=0.98)

    metrics_path = os.path.join(args.out, "metrics.csv")
    with open(metrics_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "step",
                "train_loss",      # base MSE (for comparability)
                "loss_ema",        # EMA of base MSE
                "val_loss",
                "grad_norm",
                "total_loss",      # base + bm_reg_weight * bm_reg (if enabled)
                "bm_reg",          # the unscaled regularizer value
                "d_bm",
                "d_bm_q",
                "d_pre_marg",
                "s_std",
                "lr",
            ]
        )

        for step in range(args.steps):
            if step == args.shock_step:
                for g in opt.param_groups:
                    g["lr"] = g["lr"] * args.shock_lr_mult

            idx = np.random.randint(0, x_train.shape[0], size=(args.batch,))
            xb = torch.from_numpy(x_train[idx]).to(device)
            yb = torch.from_numpy(y_train[idx]).to(device)

            opt.zero_grad(set_to_none=True)
            yhat, h = model(xb)

            base_loss = loss_fn(yhat, yb)

            bm_reg_t = None
            if args.bm_reg_weight > 0.0:
                S = build_score_matrix(h, normalize_features=args.normalize_features)
                use_q = args.bm_reg_variant == "d_bm_q"
                bm_reg_t = d_bm_from_scores(
                    S,
                    tau=args.tau,
                    n_sinkhorn=args.sinkhorn_iters,
                    global_normalize=use_q,
                )
                total_loss = base_loss + args.bm_reg_weight * bm_reg_t
            else:
                total_loss = base_loss

            total_loss.backward()
            gnorm = grad_norm(model)
            opt.step()

            # --- monitoring (no_grad, as before) ---
            with torch.no_grad():
                S_mon = build_score_matrix(h, normalize_features=args.normalize_features)

                d_bm = float(
                    d_bm_from_scores(S_mon, tau=args.tau, n_sinkhorn=args.sinkhorn_iters).cpu()
                )
                d_bm_q = float(
                    d_bm_from_scores(
                        S_mon,
                        tau=args.tau,
                        n_sinkhorn=args.sinkhorn_iters,
                        global_normalize=True,
                    ).cpu()
                )

                if args.log_pre_marg:
                    Smax = torch.max(S_mon)
                    K = torch.exp((S_mon - Smax) / max(args.tau, 1e-12))
                    Q = K / (K.sum() + 1e-12)
                    m = Q.shape[0]
                    r = torch.full((m,), 1.0 / m, device=Q.device, dtype=Q.dtype)
                    c = torch.full((m,), 1.0 / m, device=Q.device, dtype=Q.dtype)
                    dr = torch.linalg.norm(Q.sum(dim=1) - r, ord=1)
                    dc = torch.linalg.norm(Q.sum(dim=0) - c, ord=1)
                    d_pre_marg = float(0.5 * (dr + dc))
                else:
                    d_pre_marg = float("nan")

                s_std = float(S_mon.std().cpu())

                base_loss_val = float(base_loss.detach().cpu())
                loss_ema = ema.update(base_loss_val)

                if step % 50 == 0:
                    xv = torch.from_numpy(x_val).to(device)
                    yv = torch.from_numpy(y_val).to(device)
                    yv_hat, _ = model(xv)
                    vloss = float(loss_fn(yv_hat, yv).detach().cpu())
                else:
                    vloss = float("nan")

                lr = float(opt.param_groups[0]["lr"])

                total_loss_val = float(total_loss.detach().cpu())
                bm_reg_val = float(bm_reg_t.detach().cpu()) if bm_reg_t is not None else float("nan")

            w.writerow(
                [
                    step,
                    base_loss_val,
                    loss_ema,
                    vloss,
                    gnorm,
                    total_loss_val,
                    bm_reg_val,
                    d_bm,
                    d_bm_q,
                    d_pre_marg,
                    s_std,
                    lr,
                ]
            )

    print(f"Wrote metrics: {metrics_path}")


if __name__ == "__main__":
    main()