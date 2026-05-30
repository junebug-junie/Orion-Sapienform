from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ManifoldBatchMetrics:
    """Cheap geometry probes for latent-semantic-manifold training runs.

    These metrics are intentionally computed from logits / hidden states so they can
    run during normal LM training without a separate manifold-estimation pass.
    """

    loss_ce: float
    margin_mean: float
    margin_p05: float
    gap_frac: float
    entropy_mean: float
    hidden_norm_mean: float
    tangent_energy: float
    fisher_trace_proxy: float


def token_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Return per-position softmax entropy H[p(.|h)]."""

    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    return -(probs * log_probs).sum(dim=-1)


def voronoi_margin(logits: torch.Tensor) -> torch.Tensor:
    """Top-1 minus top-2 logit margin m(h)."""

    top2 = torch.topk(logits, k=2, dim=-1).values
    return top2[..., 0] - top2[..., 1]


def normalized_expressibility_gap(logits: torch.Tensor, epsilon: float) -> torch.Tensor:
    """Empirical eta(epsilon): fraction of states whose margin is below epsilon."""

    return (voronoi_margin(logits) < epsilon).float().mean()


def fisher_trace_proxy(logits: torch.Tensor) -> torch.Tensor:
    """Cheap proxy for tr(Sigma_p) = 1 - ||p||_2^2.

    The full Fisher matrix is W^T Sigma_p W. Computing it directly is too heavy
    for routine training logs, but tr(Sigma_p) tracks distributional spread and
    rises near Voronoi boundaries.
    """

    probs = F.softmax(logits, dim=-1)
    return (1.0 - (probs * probs).sum(dim=-1)).mean()


def tangent_energy(hidden: torch.Tensor, tangent: torch.Tensor | None) -> torch.Tensor:
    """Ratio of adapter/tangent update energy to hidden-state energy."""

    if tangent is None:
        return hidden.new_tensor(0.0)
    num = tangent.pow(2).mean()
    den = hidden.pow(2).mean().clamp_min(1e-8)
    return num / den


@torch.no_grad()
def summarize_manifold_batch(
    *,
    logits: torch.Tensor,
    hidden: torch.Tensor,
    loss_ce: torch.Tensor,
    tangent: torch.Tensor | None = None,
    gap_epsilon: float = 0.5,
) -> ManifoldBatchMetrics:
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
    """Penalize states inside the expressibility gap strip m(h) < epsilon.

    Use with a tiny coefficient. This is not meant to force overconfident logits;
    it is a controlled experiment: does gently pushing states away from Voronoi
    boundaries improve perplexity / generation, or does it damage uncertainty?
    """

    margin = voronoi_margin(logits)
    return F.relu(epsilon - margin).mean()


def entropy_floor_loss(logits: torch.Tensor, min_entropy: float = 0.0) -> torch.Tensor:
    """Optional anti-collapse guard for gap-loss experiments."""

    if min_entropy <= 0:
        return logits.new_tensor(0.0)
    entropy = token_entropy(logits)
    return F.relu(min_entropy - entropy).mean()
