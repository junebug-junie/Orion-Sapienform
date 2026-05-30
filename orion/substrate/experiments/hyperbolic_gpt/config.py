from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class HyperbolicGPTConfig:
    vocab_size: int = 50257
    block_size: int = 256
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256
    dropout: float = 0.1
    bias: bool = True

    # v1 hyperbolic attention knobs
    geo_lambda_init: float = 0.05
    curvature_init: float = 1.0
    use_learned_curvature: bool = True
    use_learned_geo_lambda: bool = True

    # v2 manifold-aware knobs. Defaults preserve v1-ish behavior unless model_v2
    # is explicitly selected.
    use_hyperbolic_attention: bool = True
    semantic_adapter_rank: int = 0
    margin_gap_loss_weight: float = 0.0
    margin_gap_epsilon: float = 0.5
    entropy_floor_loss_weight: float = 0.0
    min_entropy: float = 0.0

    # v3 / MoC knobs. These are intentionally inert for v1/v2 unless model_moc
    # is selected. "global" means one scalar per attention block; "per_head" means
    # one learned value per attention head inside each block.
    curvature_mode: str = "global"
    geo_lambda_mode: str = "global"
    moc_curvature_jitter: float = 0.0
    moc_lambda_jitter: float = 0.0
    tie_lm_head: bool = True

    def __post_init__(self) -> None:
        if self.n_embd % self.n_head != 0:
            raise ValueError(
                f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})"
            )
        if self.semantic_adapter_rank < 0:
            raise ValueError("semantic_adapter_rank must be >= 0")
        allowed_modes = {"global", "per_head"}
        if self.curvature_mode not in allowed_modes:
            raise ValueError(
                f"curvature_mode must be one of {sorted(allowed_modes)}, got {self.curvature_mode!r}"
            )
        if self.geo_lambda_mode not in allowed_modes:
            raise ValueError(
                f"geo_lambda_mode must be one of {sorted(allowed_modes)}, got {self.geo_lambda_mode!r}"
            )
        if self.moc_curvature_jitter < 0:
            raise ValueError("moc_curvature_jitter must be >= 0")
        if self.moc_lambda_jitter < 0:
            raise ValueError("moc_lambda_jitter must be >= 0")

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HyperbolicGPTConfig:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> HyperbolicGPTConfig:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)
