"""Column shape for BiometricsSummaryV1 -> orion_biometrics_summary (ROADMAP B1).

This file exists because the first attempt at it grepped the model's SOURCE TEXT for
`"measurements = Column(JSONB"` -- which still matches when the line is commented out. It
asserted nothing about the mapper, and `_write_row` (app/worker.py) filters payload keys
against the mapper's columns, so a field with no column is dropped SILENTLY rather than
raising. That is the failure this checks for, against the real table object.

Follows the `*_sql_shape.py` family, e.g. test_juniper_affective_state_sql_shape.py.
"""
from __future__ import annotations

from app.models.biometrics_summary import BiometricsSummarySQL

from orion.schemas.telemetry.biometrics import BiometricsSummaryV1


def test_every_payload_field_has_a_column() -> None:
    """A missing column means _write_row drops that field with no error at all."""
    columns = set(BiometricsSummarySQL.__table__.columns.keys())
    payload_fields = set(BiometricsSummaryV1.model_fields.keys())
    assert payload_fields <= columns, f"dropped silently by _write_row: {payload_fields - columns}"


def test_measurements_column_exists_on_the_mapper_not_just_in_the_source() -> None:
    assert "measurements" in BiometricsSummarySQL.__table__.columns


def test_measurements_column_is_jsonb_so_keys_stay_queryable() -> None:
    """A TEXT column would store the dict but make `measurements ? 'chassis_watts'` impossible,
    and the absent-is-not-zero contract is expressed as key presence."""
    col = BiometricsSummarySQL.__table__.columns["measurements"]
    assert col.type.__class__.__name__ == "JSONB"


def test_measurements_column_is_nullable_because_null_and_empty_mean_different_things() -> None:
    """NULL = producer predates the field (says nothing about the node).
    {} = a current producer measured nothing it could vouch for.
    Collapsing them lets a fleet total count a not-yet-upgraded node as 0 W measured."""
    assert BiometricsSummarySQL.__table__.columns["measurements"].nullable is True


def test_writer_boot_applies_the_column_before_serving() -> None:
    """The deploy-ordering trap: the model declaring `measurements` puts it in the INSERT, so
    against a DB without the column every biometrics-summary write raises UndefinedColumn --
    stopping ALL persistence for the table, not just the new field. `create_all` never adds
    columns, so the boot-time ALTER is what makes the deploy safe in either order."""
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "app" / "main.py").read_text()
    assert "orion_biometrics_summary" in src
    assert "ADD COLUMN IF NOT EXISTS measurements JSONB" in src
    assert "ALTER TABLE IF EXISTS orion_biometrics_summary" in src
