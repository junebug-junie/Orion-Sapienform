from __future__ import annotations

import torch


def project_to_ball(
    x: torch.Tensor, c: torch.Tensor | float, eps: float = 1e-5
) -> torch.Tensor:
    """Project ambient vectors into the open Poincaré ball."""
    c_t = torch.as_tensor(c, dtype=x.dtype, device=x.device)
    max_norm = (1.0 - eps) / torch.sqrt(c_t)
    norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True).clamp_min(eps)
    factor = torch.clamp(max_norm / norm, max=1.0)
    return x * factor


def mobius_add(
    x: torch.Tensor, y: torch.Tensor, c: torch.Tensor | float, eps: float = 1e-5
) -> torch.Tensor:
    c_t = torch.as_tensor(c, dtype=x.dtype, device=x.device)
    x2 = (x * x).sum(dim=-1, keepdim=True)
    y2 = (y * y).sum(dim=-1, keepdim=True)
    xy = (x * y).sum(dim=-1, keepdim=True)
    num = (1.0 + 2.0 * c_t * xy + c_t * y2) * x + (1.0 - c_t * x2) * y
    den = 1.0 + 2.0 * c_t * xy + c_t**2 * x2 * y2
    out = num / den.clamp_min(eps)
    return project_to_ball(out, c_t, eps=eps)


def expmap0(v: torch.Tensor, c: torch.Tensor | float, eps: float = 1e-5) -> torch.Tensor:
    """Exponential map at origin: tangent vector -> point on ball."""
    c_t = torch.as_tensor(c, dtype=v.dtype, device=v.device)
    sqrt_c = torch.sqrt(c_t)
    v_norm = torch.linalg.vector_norm(v, dim=-1, keepdim=True).clamp_min(eps)
    coef = torch.tanh(sqrt_c * 0.5 * v_norm) / (sqrt_c * v_norm)
    y = coef * v
    return project_to_ball(y, c_t, eps=eps)


def _artanh_clamped(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x = x.clamp(-1.0 + eps, 1.0 - eps)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


def pairwise_poincare_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    c: torch.Tensor | float,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    Memory-efficient all-pairs Poincaré distance (no [B,H,T,T,D] materialization).

    d_c(x, y) = arcosh(1 + 2c||x-y||^2 / ((1-c||x||^2)(1-c||y||^2))) / sqrt(c)

    x: (B, H, Tq, D), y: (B, H, Tk, D) -> (B, H, Tq, Tk), computed in fp32.
    """
    x = project_to_ball(x.float(), c, eps=eps)
    y = project_to_ball(y.float(), c, eps=eps)
    c_t = torch.as_tensor(c, dtype=x.dtype, device=x.device).float().clamp_min(eps)

    sqrt_c = torch.sqrt(c_t)

    x2 = (x * x).sum(dim=-1, keepdim=True)  # B,H,Tq,1
    y2 = (y * y).sum(dim=-1, keepdim=True).transpose(-2, -1)  # B,H,1,Tk

    xy = torch.matmul(x, y.transpose(-2, -1))  # B,H,Tq,Tk
    dist2 = torch.clamp(x2 + y2 - 2.0 * xy, min=0.0)

    denom_x = torch.clamp(1.0 - c_t * x2, min=eps)
    denom_y = torch.clamp(1.0 - c_t * y2, min=eps)

    z = 1.0 + 2.0 * c_t * dist2 / torch.clamp(denom_x * denom_y, min=eps)
    z = torch.clamp(z, min=1.0 + eps)

    return torch.acosh(z) / sqrt_c


def poincare_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    c: torch.Tensor | float,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    d_c(x, y) for matching (..., D) tensors (one pair per position; no T×T grid).
    """
    c_t = torch.as_tensor(c, dtype=torch.float32, device=x.device).clamp_min(eps)
    x_b = project_to_ball(x.float(), c_t, eps=eps)
    y_b = project_to_ball(y.float(), c_t, eps=eps)
    dist2 = ((x_b - y_b) ** 2).sum(dim=-1).clamp_min(0.0)
    x2 = (x_b * x_b).sum(dim=-1)
    y2 = (y_b * y_b).sum(dim=-1)
    z = 1.0 + 2.0 * c_t * dist2 / torch.clamp(
        (1.0 - c_t * x2) * (1.0 - c_t * y2), min=eps
    )
    z = torch.clamp(z, min=1.0 + eps)
    return torch.acosh(z) / torch.sqrt(c_t)


def poincare_distance_pairs(
    q: torch.Tensor,
    k: torch.Tensor,
    c: torch.Tensor | float,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Alias for attention pairwise distances: (B, H, Tq, D) x (B, H, Tk, D) -> (B, H, Tq, Tk)."""
    return pairwise_poincare_distance(q, k, c, eps=eps)
