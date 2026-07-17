"""Shape checks for the causal-geometry-snapshot SQL write path (no Postgres required).

Asserts the real wiring for bus -> sql-writer -> SQL read: `CausalGeometrySnapshotSQL`
is registered in `MODEL_MAP` under its own route key, keyed off kind
`causal.geometry.snapshot.v1` (both in `DEFAULT_ROUTE_MAP` and the operator-facing
`.env_example` -- the env alias REPLACES the default JSON entirely in live
deployments, so a settings-only change would be dead wiring), and column names
match `CausalGeometrySnapshotV1`'s own field names exactly (no `_json` suffix),
so the generic `_write_row()` path needs zero special-casing for this kind.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import create_engine, inspect
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

REPO_ROOT = Path(__file__).resolve().parents[3]
SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT), str(SERVICE_ROOT)]

from orion.schemas.causal_geometry import (  # noqa: E402
    CAUSAL_GEOMETRY_SNAPSHOT_SQL_COLUMNS,
    CausalGeometryDivergenceEntryV1,
    CausalGeometryEdgeV1,
    CausalGeometrySnapshotV1,
)

from app.models.causal_geometry_snapshot import CausalGeometrySnapshotSQL  # noqa: E402
from app.worker import INSERT_ONLY_MODELS, MODEL_MAP  # noqa: E402
from app.settings import DEFAULT_ROUTE_MAP, Settings  # noqa: E402

CHANNEL = "orion:causal_geometry:snapshot"

BASE_TS = datetime(2026, 7, 16, tzinfo=timezone.utc)

# Sourced directly from orion.schemas.causal_geometry -- the same shared
# constant services/orion-hub/scripts/api_routes.py's SELECT imports -- so this
# test fails if either side's column list ever drifts, not just if this file's
# own hardcoded expectation goes stale.
EXPECTED_COLUMNS = set(CAUSAL_GEOMETRY_SNAPSHOT_SQL_COLUMNS)


def _snapshot(**overrides) -> CausalGeometrySnapshotV1:
    defaults = dict(
        snapshot_id="snap-1",
        generated_at=BASE_TS,
        window_start=BASE_TS,
        window_end=BASE_TS,
        edges=[
            CausalGeometryEdgeV1(
                source_id="cap:transport",
                target_id="cap:orchestration",
                lag_sec=0,
                strength=0.9,
                significance=0.01,
                n_samples=100,
                window_start=BASE_TS,
                window_end=BASE_TS,
            )
        ],
        designed_topology_version="field_lattice.v1.test",
        divergence=[
            CausalGeometryDivergenceEntryV1(
                source_id="cap:transport",
                target_id="cap:orchestration#orchestration",
                observed_strength=0.9,
                designed_weight=0.2,
                delta=0.7,
                status="both",
            )
        ],
        insufficient_data=False,
        notes=["1 significant edge found."],
    )
    defaults.update(overrides)
    return CausalGeometrySnapshotV1(**defaults)


def _row_dict(snapshot: CausalGeometrySnapshotV1) -> dict:
    """Mirror `_write_row`'s generic filter -- no per-model derivation needed here."""
    data = snapshot.model_dump()
    mapper = inspect(CausalGeometrySnapshotSQL)
    valid_keys = {attr.key for attr in mapper.attrs}
    return {k: v for k, v in data.items() if k in valid_keys}


def test_default_route_map_points_causal_geometry_snapshot_at_its_sql_model() -> None:
    assert DEFAULT_ROUTE_MAP.get("causal.geometry.snapshot.v1") == "CausalGeometrySnapshotSQL"


def test_model_map_registers_causal_geometry_snapshot_sql_with_snapshot_schema() -> None:
    assert MODEL_MAP["CausalGeometrySnapshotSQL"] == (CausalGeometrySnapshotSQL, CausalGeometrySnapshotV1)


def test_channel_is_in_settings_default_subscribe_list() -> None:
    default_channels = Settings.model_fields["sql_writer_subscribe_channels"].default
    assert CHANNEL in default_channels


def test_channel_is_in_env_example_subscribe_channels() -> None:
    """Live deployments set SQL_WRITER_SUBSCRIBE_CHANNELS, and the Pydantic
    alias REPLACES the code default entirely -- the channel must also be in the
    operator contract or the wiring is dead in production. Being in MODEL_MAP/
    DEFAULT_ROUTE_MAP alone does nothing if the Redis subscriber never
    subscribes to the channel the message actually arrives on (this exact
    failure mode has hit this repo before, see test_drive_audit_sql_shape.py)."""
    env_example = (SERVICE_ROOT / ".env_example").read_text()
    for line in env_example.splitlines():
        if line.startswith("SQL_WRITER_SUBSCRIBE_CHANNELS="):
            assert CHANNEL in line
            return
    raise AssertionError("SQL_WRITER_SUBSCRIBE_CHANNELS not found in .env_example")


def test_env_example_route_map_json_includes_causal_geometry_snapshot() -> None:
    env_example = (SERVICE_ROOT / ".env_example").read_text()
    for line in env_example.splitlines():
        if line.startswith("SQL_WRITER_ROUTE_MAP_JSON="):
            assert '"causal.geometry.snapshot.v1":"CausalGeometrySnapshotSQL"' in line
            return
    raise AssertionError("SQL_WRITER_ROUTE_MAP_JSON not found in .env_example")


def test_column_shape_matches_pydantic_field_names_exactly() -> None:
    mapper = inspect(CausalGeometrySnapshotSQL)
    assert {attr.key for attr in mapper.attrs} == EXPECTED_COLUMNS


def test_row_dict_passes_through_with_no_special_casing() -> None:
    row = _row_dict(_snapshot())
    assert row["snapshot_id"] == "snap-1"
    assert row["insufficient_data"] is False
    assert len(row["edges"]) == 1
    assert row["edges"][0]["source_id"] == "cap:transport"
    assert len(row["divergence"]) == 1
    assert row["notes"] == ["1 significant edge found."]


def test_row_dict_constructs_causal_geometry_snapshot_sql_without_raising() -> None:
    row = CausalGeometrySnapshotSQL(**_row_dict(_snapshot()))
    assert row.snapshot_id == "snap-1"
    assert row.insufficient_data is False


def test_insert_only_membership() -> None:
    """CausalGeometrySnapshotSQL takes the INSERT_ONLY fast path, not merge().

    Every production cycle mints a fresh snapshot_id, so merge()'s PK SELECT
    would always miss -- pure per-write overhead. The fast path (add +
    duplicate-key catch) gives equivalent idempotency for these immutable rows,
    matching DriveAuditSQL's rationale exactly.
    """
    assert any(
        m.__name__ == "CausalGeometrySnapshotSQL" and getattr(m, "__tablename__", None) == "causal_geometry_snapshots"
        for m in INSERT_ONLY_MODELS
    )


def test_insert_only_redelivery_keeps_one_row_and_first_write_wins() -> None:
    engine = create_engine("sqlite://")
    CausalGeometrySnapshotSQL.__table__.create(bind=engine)
    Session = sessionmaker(bind=engine)

    def _insert_only(snapshot: CausalGeometrySnapshotV1) -> bool:
        sess = Session()
        try:
            try:
                sess.add(CausalGeometrySnapshotSQL(**_row_dict(snapshot)))
                sess.commit()
                return True
            except IntegrityError:
                sess.rollback()
                return False
        finally:
            sess.close()

    assert _insert_only(_snapshot()) is True

    sess = Session()
    try:
        first = sess.get(CausalGeometrySnapshotSQL, "snap-1")
        assert first.insufficient_data is False
    finally:
        sess.close()

    # Redelivery (same PK, different content) is skipped -- first write wins.
    assert _insert_only(_snapshot(insufficient_data=True, edges=[], notes=["different"])) is False

    sess = Session()
    try:
        rows = sess.query(CausalGeometrySnapshotSQL).all()
        assert len(rows) == 1
        assert rows[0].insufficient_data is False
    finally:
        sess.close()
