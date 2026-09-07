"""Shape checks for the CockpitTurnSightingSQL append path (no Postgres required).

CockpitHopV1 already publishes on orion:cockpit:hop. This is sql-writer as a
durable consumer of that same channel -- append-only rows keyed by
(correlation_id, seq) so Soft HUD rewind can SELECT ORDER BY seq.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker

from orion.schemas.cockpit_sighting import CockpitHopV1

from app.models.cockpit_turn_sighting import CockpitTurnSightingSQL
from app.cockpit_turn_sighting_persist import append_cockpit_hop
from app.settings import DEFAULT_ROUTE_MAP, Settings
from app.worker import MODEL_MAP

SERVICE_ROOT = Path(__file__).resolve().parents[1]
CHANNEL = "orion:cockpit:hop"

HOP_PAYLOAD = {
    "schema_version": "cockpit.hop.v1",
    "correlation_id": "corr-1",
    "seq": 0,
    "stage": "stance_decision",
    "visor_line": "stance · proceed",
    "status": "ok",
    "summary": {},
    "raw": {"disposition": "proceed"},
    "producer": "orion-hub",
}


@pytest.fixture()
def sqlite_sess():
    engine = create_engine("sqlite://")
    CockpitTurnSightingSQL.__table__.create(bind=engine)
    Session = sessionmaker(bind=engine)
    sess = Session()
    try:
        yield sess
    finally:
        sess.close()


def test_default_route_map_points_cockpit_hop_at_sighting_sql() -> None:
    assert DEFAULT_ROUTE_MAP.get("cockpit.hop.v1") == "CockpitTurnSightingSQL"


def test_model_map_registers_cockpit_turn_sighting_sql() -> None:
    assert MODEL_MAP["CockpitTurnSightingSQL"] == (CockpitTurnSightingSQL, CockpitHopV1)


def test_append_cockpit_hop_inserts_row(sqlite_sess) -> None:
    ok = append_cockpit_hop(sqlite_sess, HOP_PAYLOAD)
    assert ok is True
    fetched = (
        sqlite_sess.query(CockpitTurnSightingSQL)
        .filter_by(correlation_id="corr-1", seq=0)
        .one()
    )
    assert fetched.stage == "stance_decision"
    assert fetched.visor_line == "stance · proceed"
    assert fetched.status == "ok"
    assert fetched.raw == {"disposition": "proceed"}
    assert fetched.producer == "orion-hub"


def test_append_cockpit_hop_is_idempotent_on_conflict(sqlite_sess) -> None:
    assert append_cockpit_hop(sqlite_sess, HOP_PAYLOAD) is True
    again = dict(HOP_PAYLOAD)
    again["visor_line"] = "stance · overwritten"
    assert append_cockpit_hop(sqlite_sess, again) is False
    rows = sqlite_sess.query(CockpitTurnSightingSQL).filter_by(correlation_id="corr-1").all()
    assert len(rows) == 1
    assert rows[0].visor_line == "stance · proceed"


def test_append_without_correlation_id_is_a_noop(sqlite_sess) -> None:
    payload = dict(HOP_PAYLOAD)
    payload.pop("correlation_id")
    assert append_cockpit_hop(sqlite_sess, payload) is False
    assert sqlite_sess.query(CockpitTurnSightingSQL).count() == 0


def test_composite_pk_is_correlation_id_and_seq() -> None:
    pk = [col.name for col in inspect(CockpitTurnSightingSQL).primary_key]
    assert pk == ["correlation_id", "seq"]


def test_channel_is_in_settings_default_subscribe_list() -> None:
    default_channels = Settings.model_fields["sql_writer_subscribe_channels"].default
    assert CHANNEL in default_channels


def test_channel_is_in_env_example_subscribe_channels() -> None:
    env_example = (SERVICE_ROOT / ".env_example").read_text()
    for line in env_example.splitlines():
        if line.startswith("SQL_WRITER_SUBSCRIBE_CHANNELS="):
            assert CHANNEL in line
            return
    raise AssertionError("SQL_WRITER_SUBSCRIBE_CHANNELS not found in .env_example")


def test_env_example_route_map_json_includes_cockpit_hop() -> None:
    env_example = (SERVICE_ROOT / ".env_example").read_text()
    for line in env_example.splitlines():
        if line.startswith("SQL_WRITER_ROUTE_MAP_JSON="):
            assert '"cockpit.hop.v1":"CockpitTurnSightingSQL"' in line
            return
    raise AssertionError("SQL_WRITER_ROUTE_MAP_JSON not found in .env_example")


def test_generic_dispatch_does_not_contaminate_with_envelope_extras() -> None:
    worker_src = (SERVICE_ROOT / "app" / "worker.py").read_text(encoding="utf-8")
    assert "elif sql_model is CockpitTurnSightingSQL:" in worker_src
    branch = worker_src.split("elif sql_model is CockpitTurnSightingSQL:", 1)[1].split("elif ", 1)[0]
    assert "_write(sql_model, CockpitHopV1, data_to_process, {}, kind=env.kind)" in branch or (
        "_write(sql_model, None, data_to_process, {}, kind=env.kind)" in branch
    )
    assert "if sql_model_cls is CockpitTurnSightingSQL:" in worker_src
