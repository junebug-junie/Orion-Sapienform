"""Schema-level registry tests for orion-world-model (2026-08-20 scaffold).

Pins two things (same shape as test_reverie_visual_registry.py):
- request/prediction schemas are registered in *both* registry dicts
  (CLAUDE.md §6: "Do not publish unregistered event shapes").
- the round trip actually works with real nested feature-group data, not
  just an empty/placeholder payload.
"""

from __future__ import annotations

from orion.schemas.registry import SCHEMA_REGISTRY, _REGISTRY, resolve
from orion.schemas.world_model import (
    WorldModelFeatureGroupV1,
    WorldModelPredictionPayload,
    WorldModelTaskRequestPayload,
    WorldModelTrajectoryStepV1,
)

WORLD_MODEL_MESSAGE_SCHEMAS = (
    "WorldModelTaskRequestPayload",
    "WorldModelPredictionPayload",
)

WORLD_MODEL_NESTED_SCHEMAS = (
    "WorldModelFeatureGroupV1",
    "WorldModelTrajectoryStepV1",
)


def test_message_schemas_registered_in_both_registries() -> None:
    for schema_id in WORLD_MODEL_MESSAGE_SCHEMAS:
        assert schema_id in _REGISTRY
        assert schema_id in SCHEMA_REGISTRY
        assert resolve(schema_id) is _REGISTRY[schema_id]
        assert SCHEMA_REGISTRY[schema_id].model is _REGISTRY[schema_id]


def test_nested_schemas_registered_in_REGISTRY_only() -> None:
    """Same split as VisionArtifactOutputs/VisionObject in vision.py -- nested
    payload models don't carry their own bus `kind`."""
    for schema_id in WORLD_MODEL_NESTED_SCHEMAS:
        assert schema_id in _REGISTRY
        assert resolve(schema_id) is _REGISTRY[schema_id]


def test_kinds_match_channels_yaml() -> None:
    assert SCHEMA_REGISTRY["WorldModelTaskRequestPayload"].kind == "world_model.task.request"
    assert SCHEMA_REGISTRY["WorldModelPredictionPayload"].kind == "world_model.prediction"


def _sample_group(dim: int, value: float = 0.5) -> WorldModelFeatureGroupV1:
    return WorldModelFeatureGroupV1(dim=dim, vector=[value] * dim)


def _sample_step(ts: float) -> WorldModelTrajectoryStepV1:
    return WorldModelTrajectoryStepV1(
        ts=ts,
        biometrics=_sample_group(4),
        affect=_sample_group(2),
        execution_context=_sample_group(2),
        memory_pointers=_sample_group(4),
        temporal=_sample_group(1),
        vision_embedding=_sample_group(8),
    )


def test_task_request_round_trip_with_real_trajectory() -> None:
    sample = WorldModelTaskRequestPayload(
        task_type="predict_next_state",
        trajectory=[_sample_step(0.0), _sample_step(1.0)],
        meta={"source": "test"},
    )
    validated = WorldModelTaskRequestPayload.model_validate(sample.model_dump(mode="json"))
    assert validated.task_type == "predict_next_state"
    assert len(validated.trajectory) == 2
    assert validated.trajectory[0].biometrics.vector == [0.5, 0.5, 0.5, 0.5]
    assert validated.meta == {"source": "test"}


def test_prediction_payload_round_trip_and_defaults_untrained_true() -> None:
    sample = WorldModelPredictionPayload(
        ok=True,
        task_type="predict_next_state",
        device="cpu",
        state_dim=16,
        window_len=3,
        predicted_mean=[0.1] * 16,
        predicted_log_var=[0.2] * 16,
        model_param_count=154_661_376,
        timings={"inference_s": 0.01},
    )
    assert sample.model_untrained is True
    validated = WorldModelPredictionPayload.model_validate(sample.model_dump(mode="json"))
    assert validated.ok is True
    assert validated.model_untrained is True
    assert validated.model_param_count == 154_661_376
    assert len(validated.predicted_mean) == 16
