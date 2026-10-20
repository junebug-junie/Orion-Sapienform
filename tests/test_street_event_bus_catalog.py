"""StreetEventV1 contract (spec 2026-10-20 street continuity, section 6):
cataloged channel, both schema registries, and the field shape the handoff
reducer and orion-substrate-runtime PRs build on."""

from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from orion.schemas.registry import SCHEMA_REGISTRY, resolve
from orion.schemas.vision import StreetEventV1

ROOT = Path(__file__).resolve().parents[1]
CHANNEL = "orion:vision:street:events"


def _channels() -> dict[str, dict]:
    doc = yaml.safe_load((ROOT / "orion/bus/channels.yaml").read_text(encoding="utf-8")) or {}
    return {e["name"]: e for e in doc.get("channels") or [] if isinstance(e, dict) and "name" in e}


def test_street_event_channel_cataloged() -> None:
    entry = _channels()[CHANNEL]
    assert entry["schema_id"] == "StreetEventV1"
    assert entry["message_kind"] == "vision.street.event.v1"
    assert entry["producer_services"] == ["orion-vision-retina"]
    assert "orion-substrate-runtime" in entry["consumer_services"]


def test_retina_is_a_cataloged_vision_events_producer() -> None:
    assert "orion-vision-retina" in _channels()["orion:vision:events"]["producer_services"]


def test_street_event_registered_in_both_registries() -> None:
    assert resolve("StreetEventV1") is StreetEventV1
    reg = SCHEMA_REGISTRY["StreetEventV1"]
    assert reg.model is StreetEventV1 and reg.kind == "vision.street.event.v1"


def _base(**kw):
    d = dict(
        event_id="e1",
        kind="doorbell_ring",
        stream_ids=["doorbell"],
        occurred_at=datetime(2026, 10, 20, tzinfo=timezone.utc),
        salience=1.0,
        correlation_id="c1",
    )
    d.update(kw)
    return d


def test_street_event_fields() -> None:
    ev = StreetEventV1(**_base(p_value=0.03, individual_id="ind-1", person_guess="a neighbour"))
    assert ev.subject_kind == "unknown" and ev.evidence_refs == []
    assert set(StreetEventV1.model_fields) >= {
        "event_id", "kind", "stream_ids", "occurred_at", "subject_kind", "individual_id",
        "person_guess", "salience", "p_value", "evidence_refs", "correlation_id",
    }


@pytest.mark.parametrize(
    "bad",
    [dict(kind="handoff_met_known"), dict(salience=1.5), dict(p_value=-0.1), dict(stream_ids=[]), dict(extra=1)],
)
def test_street_event_rejects(bad) -> None:
    with pytest.raises(ValidationError):
        StreetEventV1(**_base(**bad))
