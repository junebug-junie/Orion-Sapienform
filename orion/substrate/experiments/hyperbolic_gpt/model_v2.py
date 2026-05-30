from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from orion.substrate.experiments.hyperbolic_gpt.config import HyperbolicGPTConfig
from orion.substrate.experiments.hyperbolic_gpt.hyperbolic import (
    expmap0,
    poincare_distance_pairs,
)
from orion.substrate.experiments.hyperbolic_gpt.manifold_metrics import (
    entropy_floor_loss,
    margin_gap_loss,
)


def _inverse_softplus(y: float) -> float:
    return math.log(math.expm1(y))


class SemanticTangentAdapter(nn.Module):
    """Low-rank residual adapter interpreted as a learned semantic tangent field.

    v2 keeps the main GPT residual stream Euclidean for stability, but injects a
    bounded low-rank update before the prediction head. This gives the experiment
    an explicit latent-manifold/tangent component without replacing every block
    with expensive Riemannian operations.
    """

    def __init__(self, n_embd: int, rank: int, dropout: float) -> None:
        super().__init__()
        self.rank = rank
        if rank <= 0:
            self.down = None
            self.up = None
        else:
            self.down = nn.Linear(n_embd, rank, bias=False)
            self.up = nn.Linear(rank, n_embd, bias=False)
            nn.init.zeros_(self.up.weight)
        self.dropout = nn.Dropout(dropout)
        self.gate_raw = nn.Parameter(torch.tensor(_inverse_softplus(0.01)))

    @property
    def gate(self) -> torch.Tensor:
        return F.softplus(self.gate_raw)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.down is None or self.up is None:
            return x, None
        tangent = self.up(F.gelu(self.down(x)))
        tangent = self.dropout(tangent)
        return x + self.gate * tangent, tangent


class HyperbolicCausalSelfAttentionV2(nn.Module):
    """Causal attention with optional hyperbolic distance bias.

    v1 always computed the expensive Poincare pair tensor. v2 makes that path
    switchable so we can run Euclidean baselines, hyperbolic attention, and
    manifold-head experiments from one class.
    """

    def __init__(self, config: HyperbolicGPTConfig) -> None:
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError(
                f"n_embd ({config.n_embd}) must be divisible by n_head ({config.n_head})"
            )
        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.n_embd = config.n_embd
        self.use_hyperbolic_attention = getattr(config, "use_hyperbolic_attention", True)

        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.q_geo = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.k_geo = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.dropout = nn.Dropout(config.dropout)
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
            persistent=False,
        )

        lam_init = _inverse_softplus(config.geo_lambda_init)
        curv_init = _inverse_softplus(config.curvature_init)
        if config.use_learned_geo_lambda:
            self.lambda_raw = nn.Parameter(torch.tensor(lam_init))
        else:
            self.register_buffer("lambda_raw", torch.tensor(lam_init))
        if config.use_learned_curvature:
            self.curvature_raw = nn.Parameter(torch.tensor(curv_init))
        else:
            self.register_buffer("curvature_raw", torch.tensor(curv_init))

    @property
    def geo_lambda(self) -> torch.Tensor:
        return F.softplus(self.lambda_raw)

    @property
    def curvature(self) -> torch.Tensor:
        return F.softplus(self.curvature_raw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        H, D = self.n_head, self.head_dim

        q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)
        k = self.k_proj(x).view(B, T, H, D).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, D).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(D))

        if self.use_hyperbolic_attention:
            qg = self.q_geo(x).view(B, T, H, D).transpose(1, 2)
            kg = self.k_geo(x).view(B, T, H, D).transpose(1, 2)
            c = self.curvature
            lam = self.geo_lambda

            with torch.autocast(device_type=x.device.type, enabled=False):
                qh = expmap0(qg.float(), c, eps=1e-5)
                kh = expmap0(kg.float(), c, eps=1e-5)
                dist = poincare_distance_pairs(qh, kh, c, eps=1e-5)
            att = att - lam * dist.to(dtype=att.dtype)

        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(y)


class MLP(nn.Module):
    def __init__(self, config: HyperbolicGPTConfig) -> None:
        super().__init__()
        self.fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = F.gelu(x)
        x = self.proj(x)
        return self.dropout(x)


class BlockV2(nn.Module):
    def __init__(self, config: HyperbolicGPTConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = HyperbolicCausalSelfAttentionV2(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class HyperbolicGPTV2(nn.Module):
    """Manifold-aware HyperbolicGPT v2.

    Output contract:
      logits, loss, aux

    aux contains hidden/tangent tensors only when requested, so standard training
    can stay close to v1 while eval/diagnostics can inspect geometry.
    """

    def __init__(self, config: HyperbolicGPTConfig) -> None:
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([BlockV2(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )
        rank = getattr(config, "semantic_adapter_rank", 0)
        self.semantic_adapter = SemanticTangentAdapter(
            config.n_embd, rank=rank, dropout=config.dropout
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        *,
        return_aux: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, torch.Tensor]]:
        B, T = idx.shape
        if T > self.config.block_size:
            raise ValueError(f"Sequence length {T} > block_size {self.config.block_size}")
        pos = torch.arange(0, T, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)
        x = self.transformer.drop(x)
        for block in self.transformer.h:
            x = block(x)
        hidden = self.transformer.ln_f(x)
        adapted, tangent = self.semantic_adapter(hidden)
        logits = self.lm_head(adapted)

        loss = None
        aux: dict[str, torch.Tensor] = {}
        if targets is not None:
            loss_ce = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss = loss_ce
            gap_weight = getattr(self.config, "margin_gap_loss_weight", 0.0)
            if gap_weight > 0:
                gap_eps = getattr(self.config, "margin_gap_epsilon", 0.5)
                loss = loss + gap_weight * margin_gap_loss(logits, epsilon=gap_eps)
            entropy_weight = getattr(self.config, "entropy_floor_loss_weight", 0.0)
            if entropy_weight > 0:
                min_entropy = getattr(self.config, "min_entropy", 0.0)
                loss = loss + entropy_weight * entropy_floor_loss(
                    logits, min_entropy=min_entropy
                )
            if return_aux:
                aux["loss_ce"] = loss_ce.detach()
        if return_aux:
            aux["hidden"] = hidden.detach()
            if tangent is not None:
                aux["tangent"] = tangent.detach()
            aux["semantic_adapter_gate"] = self.semantic_adapter.gate.detach()
        return logits, loss, aux

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size :]
            logits, _, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-8)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx
