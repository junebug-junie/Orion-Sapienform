from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from orion.substrate.experiments.hyperbolic_gpt.config import HyperbolicGPTConfig
from orion.substrate.experiments.hyperbolic_gpt.hyperbolic import expmap0, poincare_distance_pairs
from orion.substrate.experiments.hyperbolic_gpt.manifold_metrics import (
    entropy_floor_loss,
    margin_gap_loss,
)
from orion.substrate.experiments.hyperbolic_gpt.model_v2 import SemanticTangentAdapter


def _inverse_softplus(y: float) -> float:
    return math.log(math.expm1(y))


def _jittered_raw_values(init: float, count: int, jitter: float) -> torch.Tensor:
    """Create inverse-softplus raw parameters with optional multiplicative jitter."""
    if count <= 0:
        raise ValueError("count must be positive")
    base = torch.full((count,), float(init), dtype=torch.float32)
    if jitter > 0:
        # Deterministic spread instead of random init keeps smoke tests repeatable.
        spread = torch.linspace(-jitter, jitter, count, dtype=torch.float32)
        base = (base * (1.0 + spread)).clamp_min(1e-5)
    return torch.log(torch.expm1(base))


class HyperbolicCausalSelfAttentionMoC(nn.Module):
    """Mixture-of-curvature causal self-attention.

    v2 learns one curvature/lambda scalar per block. MoC can instead learn one
    curvature/lambda per attention head, making curvature an inspectable degree
    of freedom across the model. Mabrok-style diagnostics can then ask whether
    per-head curvature changes expressibility gaps, margins, entropy, and loss.
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
        self.curvature_mode = getattr(config, "curvature_mode", "global")
        self.geo_lambda_mode = getattr(config, "geo_lambda_mode", "global")

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

        lam_count = config.n_head if self.geo_lambda_mode == "per_head" else 1
        curv_count = config.n_head if self.curvature_mode == "per_head" else 1
        lam_raw = _jittered_raw_values(
            config.geo_lambda_init, lam_count, getattr(config, "moc_lambda_jitter", 0.0)
        )
        curv_raw = _jittered_raw_values(
            config.curvature_init,
            curv_count,
            getattr(config, "moc_curvature_jitter", 0.0),
        )
        if config.use_learned_geo_lambda:
            self.lambda_raw = nn.Parameter(lam_raw)
        else:
            self.register_buffer("lambda_raw", lam_raw)
        if config.use_learned_curvature:
            self.curvature_raw = nn.Parameter(curv_raw)
        else:
            self.register_buffer("curvature_raw", curv_raw)

    @property
    def geo_lambda(self) -> torch.Tensor:
        return F.softplus(self.lambda_raw)

    @property
    def curvature(self) -> torch.Tensor:
        return F.softplus(self.curvature_raw)

    def _head_values(self, values: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == "per_head":
            return values.view(1, self.n_head, 1, 1)
        return values.reshape(1, 1, 1, 1)

    @torch.no_grad()
    def geometry_diagnostics(self) -> dict[str, float]:
        c = self.curvature.detach().float().flatten()
        lam = self.geo_lambda.detach().float().flatten()
        return {
            "curvature_mean": float(c.mean().item()),
            "curvature_min": float(c.min().item()),
            "curvature_max": float(c.max().item()),
            "curvature_std": float(c.std(unbiased=False).item()) if c.numel() > 1 else 0.0,
            "geo_lambda_mean": float(lam.mean().item()),
            "geo_lambda_min": float(lam.min().item()),
            "geo_lambda_max": float(lam.max().item()),
            "geo_lambda_std": float(lam.std(unbiased=False).item()) if lam.numel() > 1 else 0.0,
        }

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
            c = self._head_values(self.curvature, self.curvature_mode)
            lam = self._head_values(self.geo_lambda, self.geo_lambda_mode)

            with torch.autocast(device_type=x.device.type, enabled=False):
                qh = expmap0(qg.float(), c.float(), eps=1e-5)
                kh = expmap0(kg.float(), c.float(), eps=1e-5)
                dist = poincare_distance_pairs(qh, kh, c.float(), eps=1e-5)
            att = att - lam.to(dtype=att.dtype) * dist.to(dtype=att.dtype)

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


class BlockMoC(nn.Module):
    def __init__(self, config: HyperbolicGPTConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = HyperbolicCausalSelfAttentionMoC(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class HyperbolicGPTMoC(nn.Module):
    """HyperbolicGPT v3: MoC architecture plus v2 manifold diagnostics."""

    def __init__(self, config: HyperbolicGPTConfig) -> None:
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([BlockMoC(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )
        rank = getattr(config, "semantic_adapter_rank", 0)
        self.semantic_adapter = SemanticTangentAdapter(
            config.n_embd, rank=rank, dropout=config.dropout
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        if getattr(config, "tie_lm_head", True):
            self.transformer.wte.weight = self.lm_head.weight

    @torch.no_grad()
    def geometry_diagnostics(self) -> dict[str, float]:
        vals: dict[str, list[float]] = {}
        for i, block in enumerate(self.transformer.h):
            d = block.attn.geometry_diagnostics()
            for k, v in d.items():
                vals.setdefault(k, []).append(v)
                vals[f"layer_{i:02d}_{k}"] = [v]
        out: dict[str, float] = {}
        for k, xs in vals.items():
            if k.startswith("layer_"):
                out[k] = xs[0]
            else:
                t = torch.tensor(xs, dtype=torch.float32)
                out[f"moc_{k}"] = float(t.mean().item())
        return out

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
        eos_token_id: int | None = None,
        greedy: bool = False,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size :]
            logits, _, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-8)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            if greedy:
                next_id = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
            if eos_token_id is not None and bool((next_id == eos_token_id).all().item()):
                break
        return idx
