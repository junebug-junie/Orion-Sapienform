"""Compile-time shape checks for the vision-events SQL write path (no Postgres required).

Regression coverage for the bug where `orion-vision-scribe` published a
non-existent `SqlWriteRequest(table="vision_events", data={...})` shape to
`orion:collapse:sql-write`. That kind (`sql.write.request`) never matched
`DEFAULT_ROUTE_MAP`, so writes silently fell through to `BusFallbackLog`
instead of landing in `vision_events`. This test asserts the real wiring:
`VisionEventSQL` is registered in `MODEL_MAP` under the `VisionEventSQL`
route key, keyed off kind `vision.event.v1`, and every field on the real
`VisionEventBundleItem` producer schema maps onto a real column on
`VisionEventSQL`.
"""
from __future__ import annotations

import sys
from pathlib import Path

from sqlalchemy import inspect

REPO_ROOT = Path(__file__).resolve().parents[3]
SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT), str(SERVICE_ROOT)]

from orion.schemas.vision import VisionEventBundleItem  # noqa: E402

from app.models.vision_event import VisionEventSQL  # noqa: E402
from app.worker import MODEL_MAP  # noqa: E402
from app.settings import DEFAULT_ROUTE_MAP  # noqa: E402


def _make_event(**overrides) -> VisionEventBundleItem:
    defaults = dict(
        event_id="evt-vision-123",
        event_type="presence",
        narrative="A person walked into the kitchen carrying a bag of groceries.",
        entities=["person", "kitchen", "groceries"],
        tags=["household"],
        confidence=0.9,
        salience=0.5,
        evidence_refs=["artifact-1"],
    )
    defaults.update(overrides)
    return VisionEventBundleItem(**defaults)


def test_default_route_map_points_vision_event_v1_at_vision_event_sql() -> None:
    assert DEFAULT_ROUTE_MAP.get("vision.event.v1") == "VisionEventSQL"


def test_model_map_registers_vision_event_sql_with_vision_event_bundle_item() -> None:
    assert MODEL_MAP["VisionEventSQL"] == (VisionEventSQL, VisionEventBundleItem)


def test_vision_event_bundle_item_fields_map_onto_real_columns() -> None:
    mapper = inspect(VisionEventSQL)
    valid_keys = {attr.key for attr in mapper.attrs}

    evt = _make_event()
    data = evt.model_dump()

    missing = [field for field in data if field not in valid_keys]
    assert not missing, f"VisionEventBundleItem fields missing from VisionEventSQL columns: {missing}"


def test_vision_event_bundle_item_data_constructs_vision_event_sql_without_raising() -> None:
    evt = _make_event()
    data = evt.model_dump()

    row = VisionEventSQL(**data)

    assert row.event_id == evt.event_id
    assert row.event_type == evt.event_type
    assert row.narrative == evt.narrative
    assert row.entities == evt.entities
    assert row.tags == evt.tags
    assert row.confidence == evt.confidence
    assert row.salience == evt.salience
    assert row.evidence_refs == evt.evidence_refs
