"""Unit tests for scripts/vision_persistence_smoke.py helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.vision_persistence_smoke as smoke  # noqa: E402
from orion.schemas.vision import VisionEventBundleItem  # noqa: E402


def test_build_smoke_bundle_item_and_payload_shape():
    event_id, correlation_id, narrative = smoke.new_smoke_ids()
    item = smoke.build_smoke_bundle_item(event_id, narrative)
    payload = smoke.build_vision_event_payload([item])
    envelope = smoke.build_intake_envelope(correlation_id, payload)

    assert envelope.kind == smoke.KIND_VISION_EVENT_BUNDLE
    assert envelope.payload["events"][0]["event_id"] == event_id
    assert envelope.correlation_id == correlation_id


def test_expected_sql_row_fields_match_bundle_item():
    event_id, correlation_id, narrative = smoke.new_smoke_ids()
    item = smoke.build_smoke_bundle_item(event_id, narrative)
    expected = smoke.expected_sql_row_fields(item, correlation_id)

    assert expected["event_id"] == event_id
    assert expected["narrative"] == narrative
    assert expected["correlation_id"] == str(correlation_id)
    assert expected["entities"] == ["orion", "vision", "smoke"]


def test_rdf_ntriple_markers_cover_event_fields():
    item = smoke.build_smoke_bundle_item("evt-abc", "smoke narrative")
    markers = smoke.rdf_ntriple_markers(
        item.event_id, item.narrative, item.event_type, item.entities
    )
    assert "http://conjourney.net/event/evt-abc" in markers
    assert "smoke narrative" in markers
    assert "smoke_test" in markers
    assert "orion" in markers


def test_coerce_sql_row_uses_vision_event_sql_model():
    event_id, correlation_id, narrative = smoke.new_smoke_ids()
    item = smoke.build_smoke_bundle_item(event_id, narrative)
    row = smoke.coerce_sql_row(item, correlation_id)

    assert row.event_id == event_id
    assert row.correlation_id == str(correlation_id)
    assert row.narrative == narrative


def test_channel_constants_match_bus_catalog():
    assert smoke.CHANNEL_VISION_EVENTS == "orion:vision:events"
    assert smoke.CHANNEL_SCRIBE_PUB == "orion:vision:scribe:pub"
    assert smoke.CHANNEL_SQL_WRITE == "orion:vision:events:sql-write"
    assert smoke.CHANNEL_RDF_ENQUEUE == "orion:rdf:enqueue"
    assert smoke.KIND_VISION_EVENT_V1 == "vision.event.v1"
    assert smoke.KIND_RDF_WRITE_REQUEST == "rdf.write.request"


def test_build_rdf_write_request_accepts_ntriples_string():
    item = VisionEventBundleItem(
        event_id="evt-rdf",
        event_type="smoke_test",
        narrative="n",
        entities=["a"],
        tags=["t"],
        confidence=0.9,
        salience=0.5,
        evidence_refs=["e"],
    )
    scribe_root = REPO_ROOT / "services" / "orion-vision-scribe"
    saved_path = sys.path[:]
    try:
        sys.path.insert(0, str(scribe_root))
        from app.main import _build_event_triples  # noqa: E402

        nt = _build_event_triples(item)
    finally:
        sys.path[:] = saved_path
        for mod_name in list(sys.modules):
            if mod_name == "app" or mod_name.startswith("app."):
                sys.modules.pop(mod_name, None)

    req = smoke.build_rdf_write_request(item.event_id, nt)
    assert req.id == "evt-rdf"
    assert isinstance(req.triples, str)
