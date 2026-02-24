from __future__ import annotations

import torch


def sinkhorn(
    K: torch.Tensor,
    r: torch.Tensor,
    c: torch.Tensor,
    n_iters: int = 30,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Sinkhorn-Knopp scaling to approximately satisfy marginals.

    Args:
        K: Nonnegative matrix, shape (m, n)
        r: Target row marginals, shape (m,)
        c: Target col marginals, shape (n,)
        n_iters: Number of Sinkhorn iterations
        eps: Numerical epsilon

    Returns:
        P: Scaled matrix in BM(r,c) approximately.
    """
    if K.dim() != 2:
        raise ValueError("K must be 2D")
    m, n = K.shape
    if r.shape != (m,):
        raise ValueError(f"r must have shape ({m},)")
    if c.shape != (n,):
        raise ValueError(f"c must have shape ({n},)")

    K = K.clamp_min(0.0)

    u = torch.ones_like(r)
    v = torch.ones_like(c)

    for _ in range(n_iters):
        Ku = K @ v
        u = r / (Ku + eps)

        Kt_u = K.t() @ u
        v = c / (Kt_u + eps)

    P = (u.unsqueeze(1) * K) * v.unsqueeze(0)
    return P


def d_bm_from_scores(
    S: torch.Tensor,
    tau: float = 0.2,
    n_sinkhorn: int = 30,
    eps: float = 1e-12,
    global_normalize: bool = False,
) -> torch.Tensor:
    """Compute D_BM(S) as in the paper.

    Uses stabilized entropic transform K = exp((S - max S)/tau).

    Args:
        S: Score matrix (m, n)

    Returns:
        Scalar tensor D_BM.
    """
    if S.dim() != 2:
        raise ValueError("S must be 2D")

    m, n = S.shape
    Smax = torch.max(S)
    K = torch.exp((S - Smax) / max(tau, eps))
    if global_normalize:
        K = K / (K.sum() + eps)

    r = torch.full((m,), 1.0 / m, device=S.device, dtype=S.dtype)
    c = torch.full((n,), 1.0 / n, device=S.device, dtype=S.dtype)

    P = sinkhorn(K, r, c, n_iters=n_sinkhorn, eps=eps)

    num = torch.linalg.norm(K - P)
    den = torch.linalg.norm(K) + eps
    return num / den
