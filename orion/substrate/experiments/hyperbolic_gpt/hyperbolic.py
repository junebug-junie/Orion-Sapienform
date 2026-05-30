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


def poincare_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    c: torch.Tensor | float,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    d_c(x, y) = 2/sqrt(c) * artanh( sqrt(c) * || (-x) ⊕_c y || )
    """
    c_t = torch.as_tensor(c, dtype=x.dtype, device=x.device)
    x_b = project_to_ball(x, c_t, eps=eps)
    y_b = project_to_ball(y, c_t, eps=eps)
    diff = mobius_add(-x_b, y_b, c_t, eps=eps)
    diff_norm = torch.linalg.vector_norm(diff, dim=-1).clamp_min(eps)
    arg = (torch.sqrt(c_t) * diff_norm).clamp(max=1.0 - eps)
    return (2.0 / torch.sqrt(c_t)) * _artanh_clamped(arg, eps=eps)


def poincare_distance_pairs(
    q: torch.Tensor,
    k: torch.Tensor,
    c: torch.Tensor | float,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    Pairwise distances for attention.
    q: (B, H, Tq, D), k: (B, H, Tk, D) -> (B, H, Tq, Tk)
    """
    c_t = torch.as_tensor(c, dtype=q.dtype, device=q.device)
    qi = project_to_ball(q, c_t, eps=eps).unsqueeze(-2)  # B,H,Tq,1,D
    kj = project_to_ball(k, c_t, eps=eps).unsqueeze(-3)  # B,H,1,Tk,D
    minus_q = -qi
    diff = mobius_add(minus_q, kj, c_t, eps=eps)
    diff_norm = torch.linalg.vector_norm(diff, dim=-1).clamp_min(eps)
    arg = (torch.sqrt(c_t) * diff_norm).clamp(max=1.0 - eps)
    return (2.0 / torch.sqrt(c_t)) * _artanh_clamped(arg, eps=eps)
