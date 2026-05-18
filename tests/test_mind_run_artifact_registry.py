"""Schema registry coverage for Mind run artifacts (including synthetic Orch HTTP failure)."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest
from pydantic import ValidationError

from orion.schemas.mind.artifact import MindRunArtifactV1
from orion.schemas.registry import resolve


def test_mind_run_artifact_v1_registry_resolve() -> None:
    model = resolve("MindRunArtifactV1")
    assert model is MindRunArtifactV1


def test_synthetic_orch_timeout_artifact_round_trip() -> None:
    from orion.mind.v1 import MindHandoffBriefV1, MindRunResultV1

    mind_run_id = uuid4()
    result = MindRunResultV1(
        mind_run_id=mind_run_id,
        ok=False,
        error_code="mind_http_timeout",
        diagnostics=["Orch failed calling /v1/mind/run", "exc_type=ReadTimeout"],
        brief=MindHandoffBriefV1(
            mind_quality="error",
            mind_authorized_for_stance_skip=False,
            machine_contract={
                "mind.orch_http_failed": True,
                "mind.orch_http_error_type": "ReadTimeout",
                "mind.orch_http_timeout_sec": 45,
            },
        ),
        mind_quality="error",
        timing_ms_by_phase={"orch_mind_http_timeout_ms": 45010.0},
    )
    artifact = MindRunArtifactV1(
        mind_run_id=mind_run_id,
        correlation_id="550e8400-e29b-41d4-a716-446655440099",
        session_id="sess-synth",
        trigger="user_turn",
        ok=False,
        error_code="mind_http_timeout",
        router_profile_id="default",
        result_jsonb=result.model_dump(mode="json"),
        request_summary_jsonb={"verb": "chat_general", "mode": "brain"},
        created_at_utc=datetime.now(timezone.utc),
    )
    raw = artifact.model_dump(mode="json")
    restored = MindRunArtifactV1.model_validate(raw)
    assert restored.ok is False
    assert restored.result_jsonb["brief"]["machine_contract"]["mind.orch_http_failed"] is True


def test_chat_request_result_payload_registry() -> None:
    from orion.core.bus.bus_schemas import ChatRequestPayload, ChatResultPayload

    assert resolve("ChatRequestPayload") is ChatRequestPayload
    assert resolve("ChatResultPayload") is ChatResultPayload
