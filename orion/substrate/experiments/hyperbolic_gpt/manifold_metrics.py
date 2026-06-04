from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ManifoldBatchMetrics:
    loss_ce: float
    margin_mean: float
    margin_p05: float
    gap_frac: float
    entropy_mean: float
    hidden_norm_mean: float
    tangent_energy: float
    fisher_trace_proxy: float


def token_entropy(logits: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    return -(probs * log_probs).sum(dim=-1)


def voronoi_margin(logits: torch.Tensor) -> torch.Tensor:
    top2 = torch.topk(logits, k=2, dim=-1).values
    return top2[..., 0] - top2[..., 1]


def fisher_trace_proxy(logits: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(logits, dim=-1)
    return (1.0 - (probs * probs).sum(dim=-1)).mean()


def tangent_energy(hidden: torch.Tensor, tangent: torch.Tensor | None) -> torch.Tensor:
    if tangent is None:
        return hidden.new_tensor(0.0)
    return tangent.pow(2).mean() / hidden.pow(2).mean().clamp_min(1e-8)


@torch.no_grad()
def summarize_manifold_batch(*, logits: torch.Tensor, hidden: torch.Tensor, loss_ce: torch.Tensor, tangent: torch.Tensor | None = None, gap_epsilon: float = 0.5) -> ManifoldBatchMetrics:
    margins = voronoi_margin(logits).flatten()
    entropy = token_entropy(logits)
    return ManifoldBatchMetrics(
        loss_ce=float(loss_ce.detach().item()),
        margin_mean=float(margins.mean().item()),
        margin_p05=float(torch.quantile(margins.float(), 0.05).item()),
        gap_frac=float((margins < gap_epsilon).float().mean().item()),
        entropy_mean=float(entropy.mean().item()),
        hidden_norm_mean=float(torch.linalg.vector_norm(hidden, dim=-1).mean().item()),
        tangent_energy=float(tangent_energy(hidden, tangent).item()),
        fisher_trace_proxy=float(fisher_trace_proxy(logits).item()),
    )


def margin_gap_loss(logits: torch.Tensor, epsilon: float = 0.5) -> torch.Tensor:
    return F.relu(epsilon - voronoi_margin(logits)).mean()


def entropy_floor_loss(logits: torch.Tensor, min_entropy: float = 0.0) -> torch.Tensor:
    if min_entropy <= 0:
        return logits.new_tensor(0.0)
    return F.relu(min_entropy - token_entropy(logits)).mean()
