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


def test_no_frame_size_keeps_declared_zone_but_strips_the_vector() -> None:
    # Without a frame size the patio-overlap check cannot run: fail closed.
    rows = build_crop_rows(_obs([
        {"label": "person", "score": 0.9, "box_xyxy": [50, 700, 150, 950], "zone": "walkway", "embedding": [1.0]},
    ], w=None, h=None), ZONES)
    assert rows[0]["zone"] == "walkway" and rows[0]["embedding"] is None
    assert rows[0]["zone_no_embed"] is False


def test_box_overlapping_the_patio_edge_is_stripped_but_not_counted_as_patio() -> None:
    # Bottom-center (400, 950) is walkway; the box reaches x=300, inside the patio.
    rows = build_crop_rows(_obs([
        {"label": "person", "score": 0.9, "box_xyxy": [300, 700, 500, 950], "embedding": [1.0],
         "embedding_ref": "e", "thumb_ref": "thumb:" + "ab" * 32},
    ]), ZONES)
    assert rows[0]["zone"] == "walkway" and rows[0]["zone_no_embed"] is False
    assert rows[0]["embedding"] is None and rows[0]["embedding_ref"] is None and rows[0]["thumb_ref"] is None


def test_zones_unavailable_fails_closed_and_strips_every_embedding() -> None:
    rows = build_crop_rows(_obs([
        {"label": "person", "score": 0.9, "box_xyxy": [400, 300, 500, 600], "embedding": [1.0, 0.0],
         "embedding_ref": "e"},
    ]), None)
    assert rows[0]["embedding"] is None and rows[0]["embedding_ref"] is None


def test_zone_load_failure_is_not_cached(monkeypatch) -> None:
    import app.vision_crop_persist as vcp

    monkeypatch.setattr(vcp, "_ZONES_CACHE", None)
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("transient")
        return {"walkway": ZONES}

    monkeypatch.setattr(vcp, "load_zones", flaky)
    assert vcp._zones_by_stream() is None
    assert vcp._zones_by_stream() == {"walkway": ZONES}
    assert vcp._zones_by_stream() == {"walkway": ZONES} and calls["n"] == 2


def test_malformed_box_is_skipped_not_fatal() -> None:
    rows = build_crop_rows(_obs([
        {"label": "person", "score": 0.9, "box_xyxy": [], "embedding": [1.0]},
        {"label": "person", "score": 0.9, "box_xyxy": [400, 300, 500, 600], "embedding": [1.0]},
    ]), ZONES)
    assert [r["crop_id"] for r in rows] == ["obs-1:1"]


def test_fallback_payload_never_carries_embeddings_or_boxes() -> None:
    import json

    from app.vision_crop_persist import redact_crop_payload

    raw = _obs([{"label": "person", "score": 0.9, "box_xyxy": [50, 700, 150, 950],
                 "embedding": [0.123, 0.456], "embedding_ref": "emb://x"}]).model_dump(mode="json")
    red = json.dumps(redact_crop_payload(raw))
    assert "0.123" not in red and "emb://x" not in red and "box_xyxy" not in red and "embedding\"" not in red
    assert '"crop_count": 1' in red


def test_worker_crop_failure_writes_redacted_fallback(monkeypatch) -> None:
    import asyncio

    import app.vision_crop_persist as vcp
    import app.worker as worker
    from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef

    def boom(payload):
        raise RuntimeError('relation "vision_crop_observation" does not exist')

    captured = {}
    monkeypatch.setattr(vcp, "persist_crop_observation", boom)
    monkeypatch.setattr(worker, "_write_fallback", lambda kind, corr, payload, err: captured.update(payload=payload))
    payload = _obs([{"label": "person", "score": 0.9, "box_xyxy": [1, 2, 3, 4], "embedding": [0.777]}]).model_dump(mode="json")
    env = BaseEnvelope(kind="vision.crop.observation.v1", source=ServiceRef(name="t", version="0", node="n"),
                       payload=payload)
    asyncio.run(worker._handle_envelope_body(env))
    assert captured["payload"]["crop_count"] == 1
    assert "0.777" not in str(captured["payload"])
