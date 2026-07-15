"""Shape checks for the drive-audit SQL write path (no Postgres required).

Asserts the real wiring for bus -> sql-writer -> SQL read:
`DriveAuditSQL` is registered in `MODEL_MAP` under the `DriveAuditSQL`
route key, keyed off kind `memory.drives.audit.v1`, the channel is in the
subscribe defaults AND `.env_example` (the env alias REPLACES the default
list in live deployments, so a settings-only change is dead wiring), and the
`active_count` derivation degrades to 0 instead of raising.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker

REPO_ROOT = Path(__file__).resolve().parents[3]
SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT), str(SERVICE_ROOT)]

from orion.core.schemas.drives import ArtifactProvenance, DriveAuditV1  # noqa: E402

from app.models.drive_audit import DriveAuditSQL  # noqa: E402
from app.worker import MODEL_MAP, _apply_drive_audit_derivations  # noqa: E402
from app.settings import DEFAULT_ROUTE_MAP, Settings  # noqa: E402

CHANNEL = "orion:memory:drives:audit"

EXPECTED_COLUMNS = {
    "artifact_id",
    "subject",
    "active_count",
    "active_drives",
    "dominant_drive",
    "drive_pressures",
    "correlation_id",
    "observed_at",
    "created_at",
}


def _make_audit(**overrides) -> DriveAuditV1:
    defaults = dict(
        artifact_id="drive-audit-abc-123",
        subject="orion",
        model_layer="drives",
        entity_id="orion",
        kind="memory.drives.audit.v1",
        ts=datetime(2026, 7, 15, 12, 0, tzinfo=timezone.utc),
        correlation_id="corr-1",
        provenance=ArtifactProvenance(intake_channel=CHANNEL),
        drive_pressures={"novelty": 0.4, "coherence": 0.2},
        active_drives=["novelty", "coherence"],
        dominant_drive="novelty",
    )
    defaults.update(overrides)
    return DriveAuditV1(**defaults)


def _row_dict(audit: DriveAuditV1) -> dict:
    """Mirror `_write_row`: filter the payload dict to mapper columns, then
    apply the DriveAuditSQL per-model derivations."""
    data = audit.model_dump()
    mapper = inspect(DriveAuditSQL)
    valid_keys = {attr.key for attr in mapper.attrs}
    filtered = {k: v for k, v in data.items() if k in valid_keys}
    _apply_drive_audit_derivations(data, filtered)
    return filtered


def test_default_route_map_points_drive_audit_at_drive_audit_sql() -> None:
    assert DEFAULT_ROUTE_MAP.get("memory.drives.audit.v1") == "DriveAuditSQL"


def test_model_map_registers_drive_audit_sql_with_audit_schema() -> None:
    assert MODEL_MAP["DriveAuditSQL"] == (DriveAuditSQL, DriveAuditV1)


def test_channel_is_in_settings_default_list() -> None:
    default_channels = Settings.model_fields["sql_writer_subscribe_channels"].default
    assert CHANNEL in default_channels


def test_channel_is_in_env_example_subscribe_channels() -> None:
    """Live deployments set SQL_WRITER_SUBSCRIBE_CHANNELS, and the Pydantic
    alias REPLACES the code default entirely — the channel must also be in the
    operator contract or the wiring is dead in production (confirmed prior
    incident of exactly this failure mode)."""
    env_example = (SERVICE_ROOT / ".env_example").read_text()
    for line in env_example.splitlines():
        if line.startswith("SQL_WRITER_SUBSCRIBE_CHANNELS="):
            assert CHANNEL in line
            return
    raise AssertionError("SQL_WRITER_SUBSCRIBE_CHANNELS not found in .env_example")


def test_column_shape_is_the_slim_measurement_contract() -> None:
    mapper = inspect(DriveAuditSQL)
    assert {attr.key for attr in mapper.attrs} == EXPECTED_COLUMNS


def test_archive_fields_are_not_columns() -> None:
    mapper = inspect(DriveAuditSQL)
    cols = {attr.key for attr in mapper.attrs}
    for archive_field in ("evidence_items", "source_event_refs", "summary", "tick_attribution"):
        assert archive_field not in cols


def test_active_count_derived_from_active_drives() -> None:
    row = _row_dict(_make_audit())
    assert row["active_count"] == 2
    assert row["active_drives"] == ["novelty", "coherence"]
    assert row["dominant_drive"] == "novelty"
    assert row["subject"] == "orion"
    assert row["artifact_id"] == "drive-audit-abc-123"


def test_active_count_zero_when_active_drives_empty() -> None:
    row = _row_dict(_make_audit(active_drives=[], dominant_drive=None))
    assert row["active_count"] == 0


def test_active_count_zero_when_active_drives_missing_or_malformed() -> None:
    # Schema default (field absent on the wire) -> empty list -> 0.
    audit = _make_audit()
    data = audit.model_dump()
    data.pop("active_drives", None)
    filtered: dict = {}
    _apply_drive_audit_derivations(data, filtered)
    assert filtered["active_count"] == 0

    # Malformed (non-list) values never raise, degrade to 0.
    for malformed in (None, "novelty", 3, {"novelty": True}):
        filtered = {}
        _apply_drive_audit_derivations({"active_drives": malformed}, filtered)
        assert filtered["active_count"] == 0


def test_observed_at_maps_from_artifact_ts() -> None:
    ts = datetime(2026, 7, 15, 12, 0, tzinfo=timezone.utc)
    row = _row_dict(_make_audit(ts=ts))
    assert row["observed_at"] == ts


def test_row_dict_constructs_drive_audit_sql_without_raising() -> None:
    row = DriveAuditSQL(**_row_dict(_make_audit()))
    assert row.artifact_id == "drive-audit-abc-123"
    assert row.active_count == 2


def test_merge_redelivery_upserts_one_row_and_preserves_created_at() -> None:
    """Re-delivery of the same artifact_id must upsert (one row), not duplicate.

    Mirrors the sql-writer generic write path (`sess.merge(Model(**filtered))`)
    against in-memory SQLite. Asserts the idempotency claim in the model
    docstring and that server-defaulted `created_at` survives a re-merge that
    omits it.
    """
    engine = create_engine("sqlite://")
    DriveAuditSQL.__table__.create(bind=engine)
    Session = sessionmaker(bind=engine)

    def _merge(audit: DriveAuditV1) -> None:
        sess = Session()
        try:
            sess.merge(DriveAuditSQL(**_row_dict(audit)))
            sess.commit()
        finally:
            sess.close()

    _merge(_make_audit())

    sess = Session()
    try:
        first = sess.get(DriveAuditSQL, "drive-audit-abc-123")
        original_created_at = first.created_at
        assert first.active_count == 2
    finally:
        sess.close()
    assert original_created_at is not None

    _merge(_make_audit(active_drives=["novelty"], dominant_drive="novelty"))

    sess = Session()
    try:
        rows = sess.query(DriveAuditSQL).all()
        assert len(rows) == 1
        row = rows[0]
        assert row.active_count == 1
        assert row.active_drives == ["novelty"]
        assert row.created_at == original_created_at
    finally:
        sess.close()
