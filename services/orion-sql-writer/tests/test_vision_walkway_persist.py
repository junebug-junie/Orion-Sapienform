"""Walkway crop/unresolved persistence: routing and the patio rule (no Postgres)."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import inspect

from app.models.vision_walkway import VisionUnresolvedSQL
from app.settings import DEFAULT_ROUTE_MAP, settings
from app.vision_crop_persist import build_crop_rows
from app.worker import MODEL_MAP
from orion.schemas.vision import VisionCropObservationV1, VisionUnresolvedV1
from orion.vision.zones import Zone

PATIO = Zone(name="patio", polygon=((0.0, 0.7), (0.35, 0.7), (0.35, 1.0), (0.0, 1.0)), embed=False)
WALK = Zone(name="walkway", polygon=((0.0, 0.35), (1.0, 0.35), (1.0, 1.0), (0.0, 1.0)), embed=True, dwell_rare_sec=120)
ZONES = [PATIO, WALK]


def _obs(crops, w=1000, h=1000):
    return VisionCropObservationV1(
        observation_id="obs-1", stream_id="walkway", observed_at=datetime(2026, 9, 24, 14, tzinfo=timezone.utc),
        frame_width=w, frame_height=h, crops=crops,
    )


def test_routes_and_subscriptions() -> None:
    assert DEFAULT_ROUTE_MAP["vision.crop.observation.v1"] == "__vision_crop_observation__"
    assert DEFAULT_ROUTE_MAP["vision.unresolved.v1"] == "VisionUnresolvedSQL"
    assert MODEL_MAP["VisionUnresolvedSQL"] == (VisionUnresolvedSQL, VisionUnresolvedV1)
    subs = settings.effective_subscribe_channels
    assert "orion:vision:crops:sql-write" in subs
    assert "orion:vision:unresolved:sql-write" in subs


def test_unresolved_fields_map_onto_columns() -> None:
    cols = {a.key for a in inspect(VisionUnresolvedSQL).attrs}
    payload = VisionUnresolvedV1(unresolved_id="u1", reason="no_label", description="a shape by the mailbox")
    missing = [k for k in payload.model_dump() if k not in cols and k != "schema_version"]
    assert not missing


def test_unresolved_model_is_not_on_the_create_all_base() -> None:
    # create_all would build the table without the migration's constraints.
    from app.db import Base

    assert "vision_unresolved" not in Base.metadata.tables
    assert "vision_crop_observation" not in Base.metadata.tables


def test_crop_rows_fan_out_one_per_crop_with_stable_ids() -> None:
    rows = build_crop_rows(_obs([
        {"label": "person", "score": 0.9, "box_xyxy": [400, 300, 500, 600], "embedding": [1.0, 0.0]},
        {"label": "dog", "score": 0.8, "box_xyxy": [600, 300, 700, 600], "embedding": [0.0, 1.0]},
    ]), ZONES)
    assert [r["crop_id"] for r in rows] == ["obs-1:0", "obs-1:1"]
    assert rows[0]["zone"] == "walkway" and rows[0]["embedding"] == [1.0, 0.0]


def test_patio_box_never_keeps_an_embedding_even_if_producer_sent_one() -> None:
    # Bottom-center (100, 950) -> normalized (0.1, 0.95): inside the patio polygon.
    rows = build_crop_rows(_obs([
        {"label": "person", "score": 0.9, "box_xyxy": [50, 700, 150, 950],
         "embedding": [0.3, 0.4], "embedding_ref": "emb://x"},
    ]), ZONES)
    assert rows[0]["zone"] == "patio"
    assert rows[0]["embedding"] is None and rows[0]["embedding_ref"] is None


def test_declared_patio_zone_wins_over_computed_zone() -> None:
    rows = build_crop_rows(_obs([
        {"label": "person", "score": 0.9, "box_xyxy": [400, 300, 500, 600], "zone": "patio",
         "embedding": [0.3, 0.4]},
    ]), ZONES)
    assert rows[0]["zone"] == "patio" and rows[0]["embedding"] is None


def test_no_frame_size_falls_back_to_declared_zone() -> None:
    rows = build_crop_rows(_obs([
        {"label": "person", "score": 0.9, "box_xyxy": [50, 700, 150, 950], "zone": "walkway", "embedding": [1.0]},
    ], w=None, h=None), ZONES)
    assert rows[0]["zone"] == "walkway" and rows[0]["embedding"] == [1.0]
