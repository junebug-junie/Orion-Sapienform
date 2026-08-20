"""World-model bus contract — `services/orion-world-model` scaffold.

Encoder + dynamics-model service intended to run on host `circe`, sharing
circe's 4th V100 32GB card via MPS with a separately-designed, out-of-scope
Infinity 2B diffusion service (see
`docs/superpowers/specs/2026-08-20-reverie-visual-chain-design.md` §3, which
explicitly deferred "whatever 'world model' turns out to mean" to a later,
undefined patch — this is that patch, port/schema/channel decisions only).

**Status: untrained inference-capable scaffolding.** The dynamics model in
`services/orion-world-model/app/model.py` is a real PyTorch module that does
a genuine forward pass on real tensors, but its weights are randomly
initialized — there is no training pipeline yet. `WorldModelPredictionPayload
.model_untrained` is always `True` in this patch. Do not read a prediction
from this service as a working next-state estimate; read it only as proof
that the shape/contract/scheduling seam works end to end.

The encoder fuses six already-computed feature groups (biometrics, affect,
execution-context, memory pointers, temporal features, and a vision
embedding). It does NOT run a vision backbone itself — `vision_embedding` is
expected to already be `orion-vision-host`'s
`VisionArtifactPayload.outputs.embedding.vector`
(`orion/schemas/vision.py::VisionEmbedding`) computed elsewhere and passed in
as a plain float vector.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

# Task types this service understands. Only one exists in this patch;
# additional task types must be added here explicitly (no free-text routing).
WORLD_MODEL_TASK_TYPES = ("predict_next_state",)


class WorldModelFeatureGroupV1(BaseModel):
    """One named feature group's already-computed vector for one trajectory
    step (e.g. biometrics or vision_embedding). `dim` is documented for the
    consumer's convenience and must match `len(vector)` -- enforced by
    `services/orion-world-model/app/model.py` at forward-pass time, not by
    this schema (a schema-level length check would require dynamic group
    dims per deployment, which conflicts with this service's fixed encoder
    input shape; see README "Feature group dims are fixed at deploy time")."""

    model_config = ConfigDict(extra="forbid")
    dim: int = Field(..., ge=1)
    vector: List[float]


class WorldModelTrajectoryStepV1(BaseModel):
    """One timestep of the six required feature groups. Callers that lack a
    real value for a group at a given tick (e.g. no fresh vision embedding)
    must send a zero vector of the configured dim -- this scaffolding does
    not impute missing modalities."""

    model_config = ConfigDict(extra="forbid")
    ts: float = Field(..., description="Unix timestamp of this trajectory step.")
    biometrics: WorldModelFeatureGroupV1
    affect: WorldModelFeatureGroupV1
    execution_context: WorldModelFeatureGroupV1
    memory_pointers: WorldModelFeatureGroupV1
    temporal: WorldModelFeatureGroupV1
    vision_embedding: WorldModelFeatureGroupV1


class WorldModelTaskRequestPayload(BaseModel):
    """Request to run the encoder+dynamics forward pass over a trajectory
    window and predict the next fused state."""

    model_config = ConfigDict(extra="forbid")
    task_type: str = Field(
        default="predict_next_state",
        description=f"One of {WORLD_MODEL_TASK_TYPES!r} -- only value in this patch.",
    )
    trajectory: List[WorldModelTrajectoryStepV1] = Field(..., min_length=1)
    meta: Optional[Dict[str, Any]] = None


class WorldModelPredictionPayload(BaseModel):
    """Reply/broadcast payload. Mirrors `orion.schemas.vision.
    VisionTaskResultPayload`'s ok/error_code/timings shape so downstream
    consumers already familiar with the vision-host contract recognize this
    one."""

    # protected_namespaces=() -- `model_untrained`/`model_param_count` below
    # otherwise trigger pydantic's "model_" protected-namespace UserWarning
    # at import time (cosmetic; the fields work correctly either way, but
    # this repo has ~15+ other schemas with the same unaddressed warning --
    # silencing it here rather than adding one more instance).
    model_config = ConfigDict(extra="forbid", protected_namespaces=())
    ok: bool
    task_type: str
    device: Optional[str] = None
    state_dim: Optional[int] = None
    window_len: Optional[int] = None
    predicted_mean: Optional[List[float]] = None
    predicted_log_var: Optional[List[float]] = None
    model_untrained: bool = Field(
        default=True,
        description=(
            "Always true until a real training pipeline exists (none does as "
            "of this patch). A genuine forward pass through real, randomly-"
            "initialized weights -- not a stub -- but the prediction carries "
            "no learned signal. See module docstring."
        ),
    )
    model_param_count: Optional[int] = None
    error: Optional[str] = None
    error_code: Optional[str] = None
    timings: Optional[Dict[str, float]] = None
    meta: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional extras (e.g. warnings); omit empty.",
    )
