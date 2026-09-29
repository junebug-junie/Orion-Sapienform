from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from orion.schemas.curiosity_urgent import (
    URGENT_REQUEST_CHANNEL,
    CuriosityUrgentRequestV1,
    CuriosityUrgentSeedV1,
)
from orion.schemas.durable_run import CuriosityRunBriefV1, CuriosityTurnRequestV1
from orion.schemas.registry import SCHEMA_REGISTRY, resolve

ROOT = Path(__file__).resolve().parents[1]
INCIDENT = "0123456789abcdef0123456789abcdef"


def _seed(**overrides) -> dict:
    base = {
        "incident_id": INCIDENT,
        "question": "Why is circe/gpu2 at 91C?",
        "trigger": "manual",
        "subject": "circe/gpu2",
        "evidence": {"gpu_temp_c": 91.0, "fans": [3200, 3150]},
        "requested_at": datetime(2026, 9, 28, 12, 0, tzinfo=timezone.utc),
    }
    base.update(overrides)
    return base


def test_valid_seed_round_trips() -> None:
    seed = CuriosityUrgentSeedV1(**_seed())
    wire = seed.model_dump(mode="json")
    assert CuriosityUrgentSeedV1.model_validate(wire) == seed
    assert seed.requested_by == "hub"


def test_question_is_stripped() -> None:
    assert CuriosityUrgentSeedV1(**_seed(question="  hot?  ")).question == "hot?"


@pytest.mark.parametrize("question", ["", "   \n\t", "x" * 2001])
def test_bad_question_rejected(question: str) -> None:
    with pytest.raises(ValidationError):
        CuriosityUrgentSeedV1(**_seed(question=question))


def test_question_at_limit_accepted() -> None:
    assert len(CuriosityUrgentSeedV1(**_seed(question="x" * 2000)).question) == 2000


@pytest.mark.parametrize("incident_id", ["NOT-HEX-AT-ALL", "0123456789AB", "abc", "g" * 12, "0" * 33])
def test_bad_incident_id_rejected(incident_id: str) -> None:
    with pytest.raises(ValidationError):
        CuriosityUrgentSeedV1(**_seed(incident_id=incident_id))


def test_bad_trigger_rejected() -> None:
    with pytest.raises(ValidationError):
        CuriosityUrgentSeedV1(**_seed(trigger="panic"))


def test_long_subject_rejected() -> None:
    with pytest.raises(ValidationError):
        CuriosityUrgentSeedV1(**_seed(subject="s" * 121))


def test_long_requested_by_rejected() -> None:
    assert CuriosityUrgentSeedV1(**_seed(requested_by="r" * 64)).requested_by == "r" * 64
    with pytest.raises(ValidationError):
        CuriosityUrgentSeedV1(**_seed(requested_by="r" * 65))


def test_extra_field_rejected() -> None:
    with pytest.raises(ValidationError):
        CuriosityUrgentSeedV1(**_seed(unexpected=True))


def test_oversized_evidence_rejected() -> None:
    with pytest.raises(ValidationError):
        CuriosityUrgentSeedV1(**_seed(evidence={"blob": "x" * 32_001}))


def test_non_json_evidence_rejected() -> None:
    with pytest.raises(ValidationError):
        CuriosityUrgentSeedV1(**_seed(evidence={"bad": object()}))


def test_request_has_same_fields_as_seed() -> None:
    req = CuriosityUrgentRequestV1(**_seed(trigger="heat", requested_by="orion-hardware-watch"))
    assert isinstance(req, CuriosityUrgentSeedV1)
    assert set(CuriosityUrgentRequestV1.model_fields) == set(CuriosityUrgentSeedV1.model_fields)


def _brief(**extra) -> CuriosityRunBriefV1:
    return CuriosityRunBriefV1(prompt="p", session_id="s", timeout_sec=900.0, **extra)


def _turn(**extra) -> CuriosityTurnRequestV1:
    return CuriosityTurnRequestV1(run_id="abc123", correlation_id="c", prompt="p", timeout_sec=900.0, **extra)


@pytest.mark.parametrize("build", [_brief, _turn])
def test_urgent_absent_on_wire_when_unset(build) -> None:
    assert "urgent" not in build().model_dump(mode="json", exclude_none=True)


@pytest.mark.parametrize("build", [_brief, _turn])
def test_urgent_present_on_wire_when_set(build) -> None:
    wire = build(urgent=CuriosityUrgentSeedV1(**_seed())).model_dump(mode="json", exclude_none=True)
    assert wire["urgent"]["incident_id"] == INCIDENT
    assert type(build()).model_validate(wire).urgent.question == "Why is circe/gpu2 at 91C?"


def test_old_shape_payloads_still_validate() -> None:
    old_brief = {"prompt": "p", "session_id": "s", "timeout_sec": 8840.0, "line": "investigate"}
    assert CuriosityRunBriefV1.model_validate(old_brief).urgent is None
    old_turn = {
        "schema_version": "curiosity.turn.request.v1",
        "run_id": "abc123",
        "correlation_id": "c",
        "prompt": "p",
        "timeout_sec": 8840.0,
        "attempt": 1,
    }
    assert CuriosityTurnRequestV1.model_validate(old_turn).urgent is None


def test_urgent_request_channel_cataloged() -> None:
    raw = yaml.safe_load((ROOT / "orion/bus/channels.yaml").read_text())
    entry = {c["name"]: c for c in raw["channels"]}[URGENT_REQUEST_CHANNEL]
    assert URGENT_REQUEST_CHANNEL == "orion:curiosity:urgent:request"
    assert entry["schema_id"] == "CuriosityUrgentRequestV1"
    assert entry["message_kind"] == "curiosity.urgent.request.v1"
    assert set(entry["producer_services"]) == {"orion-hub", "orion-hardware-watch"}
    assert entry["consumer_services"] == ["orion-hub"]


def test_urgent_request_registered() -> None:
    assert SCHEMA_REGISTRY["CuriosityUrgentRequestV1"].kind == "curiosity.urgent.request.v1"
    assert resolve("CuriosityUrgentRequestV1") is CuriosityUrgentRequestV1
