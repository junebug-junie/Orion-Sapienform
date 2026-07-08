from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock
from uuid import UUID

import pytest

import app.worker as worker
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.self_state import SelfStateDimensionV1, SelfStateV1

# NOTE: handle_self_state does `SelfStateV1.model_validate(payload)` inside a
# try/except that swallows ValidationError and returns early. A malformed
# payload would therefore NOT exercise the UUID crash at all — the test would
# be testing nothing. So build a REAL, valid SelfStateV1 and dump it. The
# self_state_id is intentionally a non-UUID string (that is the bug under test);
# it lives in the payload, not in a UUID field, so it is valid here.
_NOW = datetime(2026, 7, 7, 12, 0, tzinfo=timezone.utc)
_SCORES = {
    "coherence": 1.0, "field_intensity": 1.0, "agency_readiness": 0.41,
    "execution_pressure": 0.0, "reasoning_pressure": 0.05,
    "resource_pressure": 1.0, "reliability_pressure": 1.0,
    "continuity_pressure": 0.0, "introspection_pressure": 0.0,
    "social_pressure": 0.0, "uncertainty": 0.0, "policy_pressure": 0.0,
}


def _self_state() -> SelfStateV1:
    return SelfStateV1(
        self_state_id="self.state:tick_abc:policy.v1",  # non-UUID on purpose
        generated_at=_NOW,
        source_field_tick_id="tick_abc",
        source_field_generated_at=_NOW,
        source_attention_frame_id="frame_abc",
        source_attention_generated_at=_NOW,
        overall_intensity=0.4,
        overall_confidence=0.6,
        overall_condition="steady",
        trajectory_condition="stable",
        dimensions={
            k: SelfStateDimensionV1(dimension_id=k, score=v, confidence=0.6)
            for k, v in _SCORES.items()
        },
        dominant_field_channels={
            "contract_pressure": 1.0, "catalog_drift_pressure": 1.0, "bus_health": 1.0,
        },
        dimension_trajectory={},
    )


def _self_state_payload() -> dict:
    return _self_state().model_dump(mode="json")


@pytest.mark.asyncio
async def test_handle_self_state_uuid_crash_fixed(monkeypatch) -> None:
    captured = {}

    class _Bus:
        enabled = True

        async def publish(self, channel, env):
            captured["channel"] = channel
            captured["env"] = env

    monkeypatch.setattr(worker, "_pub_bus", _Bus(), raising=False)
    monkeypatch.setattr(worker.manager, "broadcast", AsyncMock(), raising=False)

    env = BaseEnvelope(
        kind="substrate.self_state.v1",
        source=ServiceRef(name="substrate-runtime", node="athena"),
        payload=_self_state_payload(),
    )

    # Must NOT raise pydantic ValidationError on correlation_id.
    await worker.handle_self_state(env)

    snap_env = captured["env"]
    # Envelope correlation_id is coerced to a real UUID...
    assert isinstance(snap_env.correlation_id, UUID)
    # ...while the human-readable id is preserved in the payload.
    assert snap_env.payload.correlation_id == "self.state:tick_abc:policy.v1"


def test_inner_features_settings_defaults() -> None:
    from app.settings import Settings
    s = Settings()
    assert s.inner_features_enabled is True
    assert s.inner_features_version == "seed-v1"
    assert s.channel_inner_features == "orion:self:inner_features"
    assert s.phi_degenerate_streak == 20
    assert s.orion_phi_encoder_enabled is False
