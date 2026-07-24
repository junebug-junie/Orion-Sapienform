"""Shape checks for the equilibrium-service-transition SQL write path (no
Postgres required).

Asserts the real wiring for bus -> sql-writer -> SQL read:
`EquilibriumServiceTransitionSQL` is registered in `MODEL_MAP` under its own
route key, keyed off kind `equilibrium.service.transition.v1` (both in
`DEFAULT_ROUTE_MAP` and the operator-facing `.env_example` -- the env alias
REPLACES the default JSON entirely in live deployments, so a settings-only
change would be dead wiring), on channel `orion:equilibrium:transition`
(deliberately not `orion:equilibrium:snapshot` -- see that channel's own
comment in orion/bus/channels.yaml for why), and column names match
`EquilibriumServiceTransitionV1`'s own field names exactly (no `_json`
suffix), so the generic `_write_row()` path needs zero special-casing for
this kind. Mirrors test_causal_geometry_snapshot_sql_shape.py's structure.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from sqlalchemy import create_engine, inspect
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

REPO_ROOT = Path(__file__).resolve().parents[3]
SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT), str(SERVICE_ROOT)]

from orion.schemas.telemetry.system_health import EquilibriumServiceTransitionV1  # noqa: E402

from app.models.equilibrium_service_transition import EquilibriumServiceTransitionSQL  # noqa: E402
from app.worker import INSERT_ONLY_MODELS, MODEL_MAP  # noqa: E402
from app.settings import DEFAULT_ROUTE_MAP, Settings  # noqa: E402

CHANNEL = "orion:equilibrium:transition"

BASE_TS = datetime(2026, 7, 24, tzinfo=timezone.utc)


def _transition(**overrides) -> EquilibriumServiceTransitionV1:
    defaults = dict(
        transition_id="trans-1",
        service="recall",
        node="athena",
        from_status="down",
        to_status="ok",
        transitioned_at=BASE_TS,
        down_since_ts=BASE_TS - timedelta(minutes=5),
        down_duration_ms=300_000,
        source_service="orion-equilibrium-service",
        source_node="athena",
        producer_boot_id="boot-1",
    )
    defaults.update(overrides)
    return EquilibriumServiceTransitionV1(**defaults)


def _row_dict(transition: EquilibriumServiceTransitionV1) -> dict:
    """Mirror `_write_row`'s generic filter -- no per-model derivation needed here."""
    data = transition.model_dump()
    mapper = inspect(EquilibriumServiceTransitionSQL)
    valid_keys = {attr.key for attr in mapper.attrs}
    return {k: v for k, v in data.items() if k in valid_keys}


def test_default_route_map_points_equilibrium_transition_at_its_sql_model() -> None:
    assert DEFAULT_ROUTE_MAP.get("equilibrium.service.transition.v1") == "EquilibriumServiceTransitionSQL"


def test_model_map_registers_equilibrium_service_transition_sql_with_schema() -> None:
    assert MODEL_MAP["EquilibriumServiceTransitionSQL"] == (
        EquilibriumServiceTransitionSQL,
        EquilibriumServiceTransitionV1,
    )


def test_channel_is_in_settings_default_subscribe_list() -> None:
    default_channels = Settings.model_fields["sql_writer_subscribe_channels"].default
    assert CHANNEL in default_channels


def test_channel_is_in_env_example_subscribe_channels() -> None:
    """Live deployments set SQL_WRITER_SUBSCRIBE_CHANNELS, and the Pydantic
    alias REPLACES the code default entirely -- the channel must also be in
    the operator contract or the wiring is dead in production."""
    env_example = (SERVICE_ROOT / ".env_example").read_text()
    for line in env_example.splitlines():
        if line.startswith("SQL_WRITER_SUBSCRIBE_CHANNELS="):
            assert CHANNEL in line
            return
    raise AssertionError("SQL_WRITER_SUBSCRIBE_CHANNELS not found in .env_example")


def test_env_example_route_map_json_includes_equilibrium_transition() -> None:
    env_example = (SERVICE_ROOT / ".env_example").read_text()
    for line in env_example.splitlines():
        if line.startswith("SQL_WRITER_ROUTE_MAP_JSON="):
            assert (
                '"equilibrium.service.transition.v1":"EquilibriumServiceTransitionSQL"' in line
            )
            return
    raise AssertionError("SQL_WRITER_ROUTE_MAP_JSON not found in .env_example")


def test_row_dict_passes_through_with_no_special_casing() -> None:
    row = _row_dict(_transition())
    assert row["transition_id"] == "trans-1"
    assert row["service"] == "recall"
    assert row["from_status"] == "down"
    assert row["to_status"] == "ok"
    assert row["down_duration_ms"] == 300_000


def test_row_dict_constructs_equilibrium_service_transition_sql_without_raising() -> None:
    row = EquilibriumServiceTransitionSQL(**_row_dict(_transition()))
    assert row.transition_id == "trans-1"
    assert row.to_status == "ok"


def test_insert_only_membership() -> None:
    """EquilibriumServiceTransitionSQL takes the INSERT_ONLY fast path, not
    merge(). Every real transition mints a fresh transition_id, so merge()'s
    PK SELECT would always miss -- pure per-write overhead. The fast path
    (add + duplicate-key catch) gives equivalent idempotency for these
    immutable rows, matching CausalGeometrySnapshotSQL's rationale exactly.
    """
    assert any(
        m.__name__ == "EquilibriumServiceTransitionSQL"
        and getattr(m, "__tablename__", None) == "equilibrium_service_transitions"
        for m in INSERT_ONLY_MODELS
    )


def test_insert_only_redelivery_keeps_one_row_and_first_write_wins() -> None:
    engine = create_engine("sqlite://")
    EquilibriumServiceTransitionSQL.__table__.create(bind=engine)
    Session = sessionmaker(bind=engine)

    def _insert_only(transition: EquilibriumServiceTransitionV1) -> bool:
        sess = Session()
        try:
            try:
                sess.add(EquilibriumServiceTransitionSQL(**_row_dict(transition)))
                sess.commit()
                return True
            except IntegrityError:
                sess.rollback()
                return False
        finally:
            sess.close()

    assert _insert_only(_transition()) is True

    sess = Session()
    try:
        first = sess.get(EquilibriumServiceTransitionSQL, "trans-1")
        assert first.down_duration_ms == 300_000
    finally:
        sess.close()

    # Redelivery (same PK, different content) is skipped -- first write wins.
    assert _insert_only(_transition(down_duration_ms=999_999)) is False

    sess = Session()
    try:
        rows = sess.query(EquilibriumServiceTransitionSQL).all()
        assert len(rows) == 1
        assert rows[0].down_duration_ms == 300_000
    finally:
        sess.close()
