"""Contract smoke tests for dream modernization schemas."""
from __future__ import annotations

from datetime import date

from orion.schemas.telemetry.dream import (
    DreamInternalTriggerV1,
    DreamResultV1,
    DreamTriggerPayload,
)


def test_dream_trigger_payload_minimal():
    p = DreamTriggerPayload.model_validate({"mode": "lucid"})
    assert p.mode == "lucid"


def test_dream_internal_trigger_v1():
    t = DreamInternalTriggerV1.model_validate(
        {
            "trigger_id": "t-1",
            "mode": "standard",
            "profile": "dream.v1",
            "source": "scheduler",
            "reason": "nightly",
        }
    )
    assert t.trigger_id == "t-1"
    assert t.profile == "dream.v1"


def test_dream_result_v1_defaults_and_audit():
    dr = DreamResultV1(
        narrative="A corridor of mirrors.",
        tldr="Mirrors and corridors",
        themes=["continuity"],
        symbols={"mirror": "self-model"},
    )
    assert dr.profile == "dream.v1"
    assert isinstance(dr.dream_date, date)
    metrics = dr.merged_metrics_for_sql()
    assert "_dream_audit" in metrics
    assert metrics["_dream_audit"]["profile"] == "dream.v1"
