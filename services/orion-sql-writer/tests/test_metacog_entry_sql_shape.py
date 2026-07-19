"""Shape checks for the metacog-entry SQL write path (no Postgres required).

Asserts the real wiring for bus -> sql-writer -> SQL read: `MetacogEntry` is
registered in `MODEL_MAP` under its own route key, keyed off kind
`metacog.entry.v1` (both in `DEFAULT_ROUTE_MAP` and the operator-facing
`.env_example` -- the env alias REPLACES the default JSON entirely in live
deployments, so a settings-only change would be dead wiring), and the new
`orion:metacog:sql-write` channel is actually subscribed to.
"""
from __future__ import annotations

import sys
from pathlib import Path

from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker

REPO_ROOT = Path(__file__).resolve().parents[3]
SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT), str(SERVICE_ROOT)]

from orion.schemas.metacog_entry import MetacogEntryV1, MetacogProvenance  # noqa: E402

from app.models.metacog_entry import MetacogEntry  # noqa: E402
from app.worker import MODEL_MAP  # noqa: E402
from app.settings import DEFAULT_ROUTE_MAP, Settings  # noqa: E402

CHANNEL = "orion:metacog:sql-write"


def _entry(**overrides) -> MetacogEntryV1:
    defaults = dict(
        trigger_kind="relational",
        trigger_reason="repair_pressure:level=0.80:confidence=0.85",
        summary="A brief authored summary.",
        mantra="Stay grounded.",
        provenance=MetacogProvenance(source="cortex_exec.metacog_pipeline", produces="metacog_entry"),
    )
    defaults.update(overrides)
    return MetacogEntryV1(**defaults)


def _row_dict(entry: MetacogEntryV1) -> dict:
    """Mirror `_write_row`'s generic column-name filter."""
    data = entry.model_dump(mode="json")
    mapper = inspect(MetacogEntry)
    valid_keys = {attr.key for attr in mapper.attrs}
    row = {k: v for k, v in data.items() if k in valid_keys}
    row["id"] = entry.event_id
    return row


def test_default_route_map_points_metacog_entry_at_its_sql_model() -> None:
    assert DEFAULT_ROUTE_MAP.get("metacog.entry.v1") == "MetacogEntry"


def test_model_map_registers_metacog_entry_sql_with_metacog_entry_v1_schema() -> None:
    assert MODEL_MAP["MetacogEntry"] == (MetacogEntry, MetacogEntryV1)


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


def test_env_example_route_map_json_includes_metacog_entry() -> None:
    env_example = (SERVICE_ROOT / ".env_example").read_text()
    for line in env_example.splitlines():
        if line.startswith("SQL_WRITER_ROUTE_MAP_JSON="):
            assert '"metacog.entry.v1":"MetacogEntry"' in line
            return
    raise AssertionError("SQL_WRITER_ROUTE_MAP_JSON not found in .env_example")


def test_table_name_is_orion_metacog_not_v2_of_collapse_mirror() -> None:
    """Genuinely a different table, not a collapse_mirror_v2 rename."""
    assert MetacogEntry.__tablename__ == "orion_metacog"


def test_row_dict_constructs_metacog_entry_sql_without_raising() -> None:
    entry = _entry()
    row = MetacogEntry(**_row_dict(entry))
    assert row.trigger_kind == "relational"
    assert row.summary == "A brief authored summary."
    assert row.snapshot_kind == "baseline"


def test_write_and_read_roundtrip_sqlite() -> None:
    engine = create_engine("sqlite://")
    MetacogEntry.__table__.create(bind=engine)
    Session = sessionmaker(bind=engine)

    entry = _entry(
        snapshot_kind="confirmed_dense",
        is_causally_dense=True,
        tags=["relational"],
    )
    row = MetacogEntry(**_row_dict(entry))

    sess = Session()
    try:
        sess.add(row)
        sess.commit()
        fetched = sess.get(MetacogEntry, entry.event_id)
        assert fetched is not None
        assert fetched.is_causally_dense is True
        assert fetched.snapshot_kind == "confirmed_dense"
        assert fetched.provenance["source"] == "cortex_exec.metacog_pipeline"
        assert fetched.tags == ["relational"]
    finally:
        sess.close()


def test_numeric_sisters_column_does_not_exist() -> None:
    """The whole point of the real-artifact model: no self-report column."""
    mapper = inspect(MetacogEntry)
    assert "numeric_sisters" not in {attr.key for attr in mapper.attrs}


def test_severity_and_touches_columns_exist_and_do_not_get_silently_dropped() -> None:
    """Regression test: severity/touches were added to MetacogEntryV1 in a
    correction pass but the matching SQL columns were missed -- _row_dict's
    generic column-name filter silently dropped both on every real insert
    until this was caught."""
    mapper = inspect(MetacogEntry)
    valid_keys = {attr.key for attr in mapper.attrs}
    assert "severity" in valid_keys
    assert "touches" in valid_keys

    engine = create_engine("sqlite://")
    MetacogEntry.__table__.create(bind=engine)
    Session = sessionmaker(bind=engine)

    entry = _entry(severity="critical", touches=["relational", "substrate"])
    row = MetacogEntry(**_row_dict(entry))

    sess = Session()
    try:
        sess.add(row)
        sess.commit()
        fetched = sess.get(MetacogEntry, entry.event_id)
        assert fetched is not None
        assert fetched.severity == "critical"
        assert fetched.touches == ["relational", "substrate"]
    finally:
        sess.close()
