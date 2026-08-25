"""Unit tests for app/world_model_features.py -- the pure feature-assembly
half of the world-model publish tick (worker.py's
``_world_model_publish_tick``, the first real producer for
``orion:exec:request:WorldModelService``).
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SUBSTRATE_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SUBSTRATE_ROOT) not in sys.path:
    sys.path.insert(0, str(SUBSTRATE_ROOT))

from orion.schemas.world_model import WorldModelTaskRequestPayload

from app.world_model_features import (
    ExecutionContextScalars,
    WorldModelFeatureDims,
    assemble_world_model_trajectory_step,
    build_execution_context_group,
    build_vision_embedding_group,
    temporal_features,
    zero_feature_group,
)

# services/orion-world-model/app/settings.py's own WM_DIM_* defaults as of
# this patch. Hardcoded here rather than importing that service's settings
# module -- a cross-service `app` package import collides (both services'
# test dirs use the same top-level package name; see
# services/orion-world-model/tests/test_publish_test_task.py's own comment
# about this exact collision) and would violate CLAUDE.md section 5's
# service-boundary rule at test-collection time too. This is the real
# coupling settings.py's own world_model_dim_* fields document: if that
# service's defaults ever change, this literal (and settings.py's matching
# default) must be updated in the same changeset.
_WM_DIMS = WorldModelFeatureDims(
    biometrics=32,
    affect=16,
    execution_context=16,
    memory_pointers=32,
    temporal=8,
    vision_embedding=512,
)


def test_temporal_features_deterministic_and_correct_dim():
    now = datetime(2026, 8, 25, 14, 30, 0, tzinfo=timezone.utc)  # Tuesday
    started = now - timedelta(minutes=90)
    vec = temporal_features(now, dim=8, process_started_at=started)
    assert len(vec) == 8
    # hour_frac = 14.5/24 -- hand-computed sin/cos, not re-derived from the
    # function under test (CLAUDE.md "hand-compute test fixtures").
    import math

    hour_frac = 14.5 / 24.0
    assert vec[0] == math.sin(2 * math.pi * hour_frac)
    assert vec[1] == math.cos(2 * math.pi * hour_frac)
    dow_frac = 1 / 7.0  # Tuesday -- Monday=0
    assert vec[2] == math.sin(2 * math.pi * dow_frac)
    assert vec[3] == math.cos(2 * math.pi * dow_frac)
    minute_frac = 30 / 60.0
    assert vec[4] == math.sin(2 * math.pi * minute_frac)
    assert vec[5] == math.cos(2 * math.pi * minute_frac)
    # session_elapsed = tanh(5400/3600) = tanh(1.5)
    assert vec[6] == math.tanh(1.5)
    assert vec[7] == 0.0
    # Same wall-clock input twice -> identical output (pure function).
    assert temporal_features(now, dim=8, process_started_at=started) == vec


def test_temporal_features_truncates_and_pads_to_configured_dim():
    now = datetime(2026, 8, 25, 0, 0, 0, tzinfo=timezone.utc)
    truncated = temporal_features(now, dim=3, process_started_at=now)
    assert len(truncated) == 3
    padded = temporal_features(now, dim=12, process_started_at=now)
    assert len(padded) == 12
    assert padded[8:] == [0.0, 0.0, 0.0, 0.0]


def test_temporal_features_session_elapsed_never_negative_or_unbounded():
    # process_started_at AFTER now (clock skew) must not go negative or raise.
    now = datetime(2026, 8, 25, 0, 0, 0, tzinfo=timezone.utc)
    future_start = now + timedelta(hours=1)
    vec = temporal_features(now, dim=8, process_started_at=future_start)
    assert vec[6] == 0.0  # max(0.0, negative) -> tanh(0) == 0.0


def test_build_execution_context_group_populates_real_slots_only():
    scalars = ExecutionContextScalars(execution=0.4, chat=None, route=0.1, bus_synaptic=0.0)
    group, real_domains = build_execution_context_group(16, scalars)
    assert group.dim == 16
    assert len(group.vector) == 16
    assert group.vector[0] == 0.4  # execution
    assert group.vector[1] == 0.0  # chat: None -> left at 0.0
    assert group.vector[2] == 0.1  # route
    assert group.vector[3] == 0.0  # bus_synaptic: real 0.0, matches padding by coincidence
    assert group.vector[4:] == [0.0] * 12
    # chat excluded (None), bus_synaptic included (real 0.0 != "not available").
    assert real_domains == ["execution", "route", "bus_synaptic"]


def test_build_execution_context_group_all_none_is_all_zero():
    scalars = ExecutionContextScalars(execution=None, chat=None, route=None, bus_synaptic=None)
    group, real_domains = build_execution_context_group(16, scalars)
    assert group.vector == [0.0] * 16
    assert real_domains == []


def test_build_execution_context_group_handles_dim_smaller_than_domain_count():
    scalars = ExecutionContextScalars(execution=1.0, chat=1.0, route=1.0, bus_synaptic=1.0)
    group, real_domains = build_execution_context_group(2, scalars)
    assert group.dim == 2
    assert group.vector == [1.0, 1.0]
    assert real_domains == ["execution", "chat"]


def test_build_vision_embedding_group_real_vector_used_when_dim_matches():
    raw = [0.1] * 512
    group, meta = build_vision_embedding_group(512, raw_vector=raw)
    assert group.dim == 512
    assert group.vector == raw
    assert meta == {"vision_source": "real"}


def test_build_vision_embedding_group_missing_vector_zero_fills():
    group, meta = build_vision_embedding_group(512, raw_vector=None)
    assert group.vector == [0.0] * 512
    assert meta == {"vision_source": "unavailable"}


def test_build_vision_embedding_group_empty_vector_zero_fills():
    group, meta = build_vision_embedding_group(512, raw_vector=[])
    assert group.vector == [0.0] * 512
    assert meta == {"vision_source": "unavailable"}


def test_build_vision_embedding_group_dim_mismatch_zero_fills_and_flags_both_dims():
    """The defensive path: a real embedding of the WRONG length must not raise
    and must not silently pass through -- it zero-fills and names both the
    observed and configured dims so a human can fix the real number."""
    raw = [0.2] * 1152  # e.g. a real SigLIP2 so400m width, hypothetically
    group, meta = build_vision_embedding_group(512, raw_vector=raw)
    assert group.dim == 512
    assert group.vector == [0.0] * 512
    assert meta["vision_source"] == "dim_mismatch"
    assert meta["vision_dim_observed"] == 1152
    assert meta["vision_dim_configured"] == 512


def test_assemble_world_model_trajectory_step_validates_against_schema_all_zero():
    now = datetime(2026, 8, 25, 12, 0, 0, tzinfo=timezone.utc)
    step, meta = assemble_world_model_trajectory_step(
        now=now,
        process_started_at=now,
        dims=_WM_DIMS,
        execution_context=ExecutionContextScalars(None, None, None, None),
        vision_embedding_vector=None,
    )
    payload = WorldModelTaskRequestPayload(
        task_type="predict_next_state", trajectory=[step], meta=meta
    )
    # Re-validate via model_validate(model_dump()) -- the actual wire path
    # worker.py's tick takes (payload.model_dump(mode="json") -> envelope).
    revalidated = WorldModelTaskRequestPayload.model_validate(payload.model_dump(mode="json"))
    assert len(revalidated.trajectory) == 1
    step_out = revalidated.trajectory[0]
    assert step_out.biometrics.dim == 32 and len(step_out.biometrics.vector) == 32
    assert step_out.affect.dim == 16 and len(step_out.affect.vector) == 16
    assert step_out.execution_context.dim == 16 and len(step_out.execution_context.vector) == 16
    assert step_out.memory_pointers.dim == 32 and len(step_out.memory_pointers.vector) == 32
    assert step_out.temporal.dim == 8 and len(step_out.temporal.vector) == 8
    assert step_out.vision_embedding.dim == 512 and len(step_out.vision_embedding.vector) == 512
    assert set(meta["zero_filled_groups"]) == {
        "biometrics",
        "affect",
        "memory_pointers",
        "vision_embedding",
    }
    assert meta["vision_source"] == "unavailable"
    assert meta["real_execution_context_domains"] == []


def test_assemble_world_model_trajectory_step_with_real_execution_and_vision():
    now = datetime(2026, 8, 25, 12, 0, 0, tzinfo=timezone.utc)
    step, meta = assemble_world_model_trajectory_step(
        now=now,
        process_started_at=now - timedelta(seconds=120),
        dims=_WM_DIMS,
        execution_context=ExecutionContextScalars(
            execution=0.2, chat=0.0, route=0.05, bus_synaptic=0.1
        ),
        vision_embedding_vector=[0.3] * 512,
    )
    payload = WorldModelTaskRequestPayload(
        task_type="predict_next_state", trajectory=[step], meta=meta
    )
    # Must validate cleanly end to end.
    WorldModelTaskRequestPayload.model_validate(payload.model_dump(mode="json"))
    assert step.execution_context.vector[:4] == [0.2, 0.0, 0.05, 0.1]
    assert step.vision_embedding.vector == [0.3] * 512
    assert meta["real_execution_context_domains"] == ["execution", "chat", "route", "bus_synaptic"]
    assert "vision_embedding" not in meta["zero_filled_groups"]
    assert set(meta["zero_filled_groups"]) == {"biometrics", "affect", "memory_pointers"}


def test_zero_feature_group_shape():
    g = zero_feature_group(5)
    assert g.dim == 5
    assert g.vector == [0.0] * 5
