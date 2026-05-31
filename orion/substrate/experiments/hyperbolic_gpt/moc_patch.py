from __future__ import annotations

import math
from types import MethodType

import torch
import torch.nn.functional as F

from orion.substrate.experiments.hyperbolic_gpt.hyperbolic import expmap0, poincare_distance_pairs


def _raw(init: float, count: int, jitter: float) -> torch.Tensor:
    vals = torch.full((count,), float(init), dtype=torch.float32)
    if jitter > 0 and count > 1:
        vals = (vals * (1.0 + torch.linspace(-jitter, jitter, count))).clamp_min(1e-5)
    return torch.log(torch.expm1(vals))


def _head_values(attn, values: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "per_head":
        return values.view(1, attn.n_head, 1, 1)
    return values.reshape(1, 1, 1, 1)


def _moc_forward(attn, x: torch.Tensor) -> torch.Tensor:
    B, T, C = x.size()
    H, D = attn.n_head, attn.head_dim
    q = attn.q_proj(x).view(B, T, H, D).transpose(1, 2)
    k = attn.k_proj(x).view(B, T, H, D).transpose(1, 2)
    v = attn.v_proj(x).view(B, T, H, D).transpose(1, 2)
    scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(D))

    qg = attn.q_geo(x).view(B, T, H, D).transpose(1, 2)
    kg = attn.k_geo(x).view(B, T, H, D).transpose(1, 2)
    c = _head_values(attn, attn.curvature, attn.curvature_mode)
    lam = _head_values(attn, attn.geo_lambda, attn.geo_lambda_mode)
    with torch.autocast(device_type=x.device.type, enabled=False):
        qh = expmap0(qg.float(), c.float(), eps=1e-5)
        kh = expmap0(kg.float(), c.float(), eps=1e-5)
        dist = poincare_distance_pairs(qh, kh, c.float(), eps=1e-5)
    scores = scores - lam.to(dtype=scores.dtype) * dist.to(dtype=scores.dtype)
    scores = scores.masked_fill(attn.bias[:, :, :T, :T] == 0, float("-inf"))
    probs = attn.dropout(F.softmax(scores, dim=-1))
    y = probs @ v
    y = y.transpose(1, 2).contiguous().view(B, T, C)
    return attn.out_proj(y)


def _diag(attn) -> dict[str, float]:
    c = attn.curvature.detach().float().flatten()
    lam = attn.geo_lambda.detach().float().flatten()
    return {
        "curvature_mean": float(c.mean().item()),
        "curvature_std": float(c.std(unbiased=False).item()) if c.numel() > 1 else 0.0,
        "geo_lambda_mean": float(lam.mean().item()),
        "geo_lambda_std": float(lam.std(unbiased=False).item()) if lam.numel() > 1 else 0.0,
    }


def apply_moc(model, config) -> None:
    """Turn an existing HyperbolicGPT into a per-head MoC variant in-place."""
    for block in model.transformer.h:
        attn = block.attn
        attn.curvature_mode = config.curvature_mode
        attn.geo_lambda_mode = config.geo_lambda_mode
        lam_n = config.n_head if config.geo_lambda_mode == "per_head" else 1
        curv_n = config.n_head if config.curvature_mode == "per_head" else 1
        lam = _raw(config.geo_lambda_init, lam_n, config.moc_lambda_jitter).to(next(attn.parameters()).device)
        curv = _raw(config.curvature_init, curv_n, config.moc_curvature_jitter).to(next(attn.parameters()).device)
        if hasattr(attn, "lambda_raw"):
            del attn._parameters["lambda_raw"]
        if hasattr(attn, "curvature_raw"):
            del attn._parameters["curvature_raw"]
        attn.lambda_raw = torch.nn.Parameter(lam) if config.use_learned_geo_lambda else lam
        attn.curvature_raw = torch.nn.Parameter(curv) if config.use_learned_curvature else curv
        attn.forward = MethodType(_moc_forward, attn)
        attn.geometry_diagnostics = MethodType(_diag, attn)


def model_geometry_diagnostics(model) -> dict[str, float]:
    vals: dict[str, list[float]] = {}
    raw = model.module if hasattr(model, "module") else model
    for block in raw.transformer.h:
        if not hasattr(block.attn, "geometry_diagnostics"):
            continue
        for k, v in block.attn.geometry_diagnostics().items():
            vals.setdefault(k, []).append(v)
    return {f"moc_{k}": float(torch.tensor(v).mean().item()) for k, v in vals.items()}
