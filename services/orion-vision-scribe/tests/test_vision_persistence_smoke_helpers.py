"""Unit tests for scripts/vision_persistence_smoke.py helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.vision_persistence_smoke as smoke  # noqa: E402


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
    assert smoke.KIND_VISION_EVENT_V1 == "vision.event.v1"


def test_no_rdf_symbols_remain_on_smoke_module():
    """Regression for the 2026-07-23 RDF-write removal: guards against the
    RDF verification path silently reappearing on this module."""
    assert not hasattr(smoke, "CHANNEL_RDF_ENQUEUE")
    assert not hasattr(smoke, "CHANNEL_RDF_CONFIRM")
    assert not hasattr(smoke, "rdf_ntriple_markers")
    assert not hasattr(smoke, "build_rdf_write_request")
