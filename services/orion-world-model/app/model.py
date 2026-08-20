"""World-model encoder + dynamics model.

**Untrained inference-capable scaffolding** (see `orion/schemas/world_model.py`
module docstring). Both modules below do a genuine forward pass on real
tensors through real (if randomly-initialized) weights -- this is not a stub
that skips the model, and it does not return hardcoded/fake output. But there
is no training pipeline yet, so a prediction from this module carries no
learned signal. `WorldModel.num_parameters()` and every caller that surfaces
a prediction must keep `model_untrained=True` visible (see `app/main.py`).

Encoder
-------
Fuses six already-computed feature groups (biometrics, affect,
execution-context, memory pointers, temporal features, and a vision
embedding) via small per-group linear branches + a fusion layer. This is
deliberately NOT a new backbone -- vision embeddings are expected to already
be `orion-vision-host`'s `VisionArtifactPayload.outputs.embedding.vector`,
computed elsewhere.

Dynamics
--------
A causal (autoregressive-masked) Transformer encoder over a trajectory
window of fused states, predicting a next-state Gaussian (mean, log_var).
Sized (default hyperparameters in `app/settings.py`) to land at ~150M total
parameters -- a concrete default within the requested ~100-350M range, not a
spec (README "Assumptions").
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Dict, Tuple

import torch
from torch import nn

FEATURE_GROUP_NAMES = (
    "biometrics",
    "affect",
    "execution_context",
    "memory_pointers",
    "temporal",
    "vision_embedding",
)


@dataclass(frozen=True)
class FeatureGroupDims:
    biometrics: int
    affect: int
    execution_context: int
    memory_pointers: int
    temporal: int
    vision_embedding: int

    @property
    def total(self) -> int:
        return sum(getattr(self, f.name) for f in fields(self))

    def as_dict(self) -> Dict[str, int]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


class FeatureBranch(nn.Module):
    """One small per-group linear projection: Linear -> LayerNorm -> GELU."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class WorldModelEncoder(nn.Module):
    """Small MLP fusion encoder. Not a backbone -- fuses pre-computed
    feature-group vectors into one fused state vector per timestep."""

    def __init__(self, dims: FeatureGroupDims, fusion_dim: int = 512):
        super().__init__()
        self.dims = dims
        self.fusion_dim = fusion_dim
        self.branches = nn.ModuleDict(
            {name: FeatureBranch(getattr(dims, name), fusion_dim) for name in FEATURE_GROUP_NAMES}
        )
        self.fuse = nn.Sequential(
            nn.Linear(fusion_dim * len(FEATURE_GROUP_NAMES), fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
        )

    def forward(self, **groups: torch.Tensor) -> torch.Tensor:
        """`groups` keys must be exactly `FEATURE_GROUP_NAMES`; each value is
        `[batch, group_dim]`. Returns fused state `[batch, fusion_dim]`."""
        missing = set(FEATURE_GROUP_NAMES) - set(groups)
        if missing:
            raise ValueError(f"WorldModelEncoder.forward missing feature groups: {sorted(missing)}")
        parts = [self.branches[name](groups[name]) for name in FEATURE_GROUP_NAMES]
        fused = torch.cat(parts, dim=-1)
        return self.fuse(fused)


class WorldModelDynamics(nn.Module):
    """Causal Transformer-encoder dynamics model. Consumes a trajectory
    window of fused encoder states `[batch, window, fusion_dim]` and predicts
    a next-state Gaussian (mean, log_var) of dim `state_dim`."""

    def __init__(
        self,
        fusion_dim: int = 512,
        d_model: int = 1024,
        nhead: int = 16,
        num_layers: int = 12,
        dim_feedforward: int = 4096,
        max_window: int = 32,
        state_dim: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.state_dim = state_dim or fusion_dim
        self.max_window = max_window
        self.input_proj = nn.Linear(fusion_dim, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_window, d_model))
        nn.init.normal_(self.pos_embedding, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, self.state_dim * 2)  # mean + log_var

    def forward(self, trajectory: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """`trajectory`: `[batch, window, fusion_dim]`, window <= max_window."""
        batch, window, _ = trajectory.shape
        if window > self.max_window:
            raise ValueError(f"trajectory window {window} exceeds max_window {self.max_window}")
        x = self.input_proj(trajectory) + self.pos_embedding[:, :window, :]
        # Causal mask: step t only attends to steps <= t. This is a genuine
        # dynamics constraint (predicting forward from history), not
        # bidirectional leakage from future steps.
        mask = nn.Transformer.generate_square_subsequent_mask(window).to(
            device=trajectory.device, dtype=x.dtype
        )
        encoded = self.transformer(x, mask=mask)
        last = encoded[:, -1, :]  # predict next-state from the final window step
        out = self.head(last)
        mean, log_var = out.chunk(2, dim=-1)
        return mean, log_var


class WorldModel(nn.Module):
    """Encoder + dynamics, assembled. `forward` takes a dict of
    `group_name -> [batch, window, group_dim]` tensors (one full trajectory)
    and returns `(mean, log_var)` of shape `[batch, state_dim]`."""

    def __init__(self, dims: FeatureGroupDims, fusion_dim: int = 512, **dynamics_kwargs):
        super().__init__()
        self.encoder = WorldModelEncoder(dims, fusion_dim=fusion_dim)
        self.dynamics = WorldModelDynamics(fusion_dim=fusion_dim, **dynamics_kwargs)

    def forward(self, trajectory_features: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        missing = set(FEATURE_GROUP_NAMES) - set(trajectory_features)
        if missing:
            raise ValueError(f"WorldModel.forward missing feature groups: {sorted(missing)}")
        any_tensor = trajectory_features["biometrics"]
        window = any_tensor.shape[1]
        fused_steps = []
        for t in range(window):
            step = {name: trajectory_features[name][:, t, :] for name in FEATURE_GROUP_NAMES}
            fused_steps.append(self.encoder(**step))
        trajectory = torch.stack(fused_steps, dim=1)  # [batch, window, fusion_dim]
        return self.dynamics(trajectory)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
