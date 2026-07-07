from __future__ import annotations

from uuid import uuid4

import pytest

from app.mind_enrichment import publish_mind_run_artifact_for_thought
from orion.core.bus.bus_schemas import ServiceRef
from orion.mind.v1 import MindHandoffBriefV1, MindRunPolicyV1, MindRunRequestV1, MindRunResultV1
from orion.schemas.mind.artifact import MindRunArtifactV1
from orion.schemas.thought import HubAssociationBundleV1, StanceReactRequestV1


class _FakeBus:
    def __init__(self) -> None:
        self.published: list[tuple[str, object]] = []

    async def publish(self, channel: str, envelope) -> None:
        self.published.append((channel, envelope))


class _RaisingBus:
    async def publish(self, *_a, **_k) -> None:
        raise RuntimeError("bus down")


def _request() -> StanceReactRequestV1:
    return StanceReactRequestV1(
        correlation_id="corr-1",
        session_id="sess-1",
        user_message="hi",
        association=HubAssociationBundleV1(
            correlation_id="corr-1", broadcast=None, broadcast_stale=True,
            read_source="hub_sql_fallback",
        ),
        repair_bundle=None,
        stance_inputs={"user_message": "hi"},
    )


def _mind_req() -> MindRunRequestV1:
    return MindRunRequestV1(
        correlation_id="corr-1", session_id="sess-1", trigger="user_turn",
        snapshot_inputs={"user_text": "hi", "messages_tail": []},
        policy=MindRunPolicyV1(n_loops_max=1, wall_time_ms_max=12000, router_profile_id="default"),
    )


def _mind_res() -> MindRunResultV1:
    return MindRunResultV1(
        mind_run_id=uuid4(), ok=True, snapshot_hash="hash-1",
        brief=MindHandoffBriefV1(mind_quality="meaningful_synthesis"),
        mind_quality="meaningful_synthesis",
    )


@pytest.mark.asyncio
async def test_artifact_published_with_orion_mode() -> None:
    bus = _FakeBus()
    await publish_mind_run_artifact_for_thought(
        bus,
        source=ServiceRef(name="orion-thought", node="athena", version="0.1.0"),
        request=_request(), mind_req=_mind_req(), mind_res=_mind_res(),
        channel="orion:mind:artifact",
    )
    assert len(bus.published) == 1
    channel, envelope = bus.published[0]
    assert channel == "orion:mind:artifact"
    artifact = MindRunArtifactV1.model_validate(envelope.payload)
    assert artifact.request_summary_jsonb["mode"] == "orion"
    assert artifact.request_summary_jsonb["correlation_id"] == "corr-1"
    assert artifact.ok is True
    assert artifact.router_profile_id == "default"


@pytest.mark.asyncio
async def test_artifact_publish_failure_is_swallowed() -> None:
    # Publish failure must never propagate out of the stance stage.
    await publish_mind_run_artifact_for_thought(
        _RaisingBus(),
        source=ServiceRef(name="orion-thought", node="athena", version="0.1.0"),
        request=_request(), mind_req=_mind_req(), mind_res=_mind_res(),
        channel="orion:mind:artifact",
    )  # must not raise
