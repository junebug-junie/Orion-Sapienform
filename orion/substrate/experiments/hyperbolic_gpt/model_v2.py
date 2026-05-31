from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from orion.substrate.experiments.hyperbolic_gpt.config import HyperbolicGPTConfig
from orion.substrate.experiments.hyperbolic_gpt.manifold_metrics import entropy_floor_loss, margin_gap_loss
from orion.substrate.experiments.hyperbolic_gpt.model import Block


class SemanticTangentAdapter(nn.Module):
    def __init__(self, n_embd: int, rank: int = 0, dropout: float = 0.1) -> None:
        super().__init__()
        self.down = nn.Linear(n_embd, rank, bias=False) if rank > 0 else None
        self.up = nn.Linear(rank, n_embd, bias=False) if rank > 0 else None
        if self.up is not None:
            nn.init.zeros_(self.up.weight)
        self.dropout = nn.Dropout(dropout)
        self.gate_raw = nn.Parameter(torch.tensor(math.log(math.expm1(0.01))))

    @property
    def gate(self) -> torch.Tensor:
        return F.softplus(self.gate_raw)

    def forward(self, x: torch.Tensor):
        if self.down is None or self.up is None:
            return x, None
        tangent = self.dropout(self.up(F.gelu(self.down(x))))
        return x + self.gate * tangent, tangent


class HyperbolicGPTV2(nn.Module):
    def __init__(self, config: HyperbolicGPTConfig) -> None:
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )
        self.semantic_adapter = SemanticTangentAdapter(config.n_embd, config.semantic_adapter_rank, config.dropout)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        if config.tie_lm_head:
            self.transformer.wte.weight = self.lm_head.weight

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None, *, return_aux: bool = False):
        _, T = idx.shape
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
        aux = {}
        if targets is not None:
            loss_ce = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss = loss_ce
            if self.config.margin_gap_loss_weight > 0:
                loss = loss + self.config.margin_gap_loss_weight * margin_gap_loss(logits, self.config.margin_gap_epsilon)
            if self.config.entropy_floor_loss_weight > 0:
                loss = loss + self.config.entropy_floor_loss_weight * entropy_floor_loss(logits, self.config.min_entropy)
            aux["loss_ce"] = loss_ce.detach()
        if return_aux:
            aux["hidden"] = hidden.detach()
            if tangent is not None:
                aux["tangent"] = tangent.detach()
            aux["semantic_adapter_gate"] = self.semantic_adapter.gate.detach()
        return logits, loss, aux
