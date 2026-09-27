"""Newest-wins upserts for the energy tables.

A late utility correction, a re-priced cycle, or a run whose house share arrived a
day later all re-deliver the same natural key. The WHERE clause keeps a stale
redelivery (bus replay, out-of-order handler) from overwriting fresher data.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Callable

from sqlalchemy import Date, DateTime, inspect
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from app.models.energy import (
    EnergyBillActualSQL,
    EnergyBillForecastSQL,
    EnergyCostAccruedSQL,
    EnergyImporterStatusSQL,
    EnergyReconcileSQL,
    EnergyRunCostSQL,
    EnergyStakesSnapshotSQL,
    EnergyUsageIntervalSQL,
)


def _columns(model: type, data: dict[str, Any]) -> dict[str, Any]:
    # Coerce by the column's SQL type, not its name: billing_period_start is a DATE.
    types = {col.key: col.type for col in inspect(model).columns if col.key != "id"}
    out = {k: v for k, v in data.items() if k in types}
    for key, value in list(out.items()):
        if not isinstance(value, str):
            continue
        if isinstance(types[key], DateTime):
            out[key] = datetime.fromisoformat(value.replace("Z", "+00:00"))
        elif isinstance(types[key], Date):
            out[key] = date.fromisoformat(value)
    return out


def _upsert(
    sess: Session, model: type, data: dict[str, Any], *, conflict: dict[str, Any], newer_col: str
) -> bool:
    values = _columns(model, data)
    stmt = insert(model).values(**values)
    table = model.__table__
    immutable = set(conflict.get("index_elements") or []) | {"id"}
    stmt = stmt.on_conflict_do_update(
        **conflict,
        set_={k: stmt.excluded[k] for k in values if k not in immutable},
        where=table.c[newer_col] <= stmt.excluded[newer_col],
    )
    sess.execute(stmt)
    sess.commit()
    return True


def _insert_once(sess: Session, model: type, data: dict[str, Any], *, constraint: str) -> bool:
    stmt = insert(model).values(**_columns(model, data)).on_conflict_do_nothing(constraint=constraint)
    sess.execute(stmt)
    sess.commit()
    return True


def upsert_energy_usage_interval(sess: Session, data: dict[str, Any]) -> bool:
    return _upsert(
        sess, EnergyUsageIntervalSQL, data,
        conflict={"constraint": "uq_energy_usage_interval_point_start"}, newer_col="retrieved_at",
    )


def upsert_energy_cost_accrued(sess: Session, data: dict[str, Any]) -> bool:
    return _upsert(
        sess, EnergyCostAccruedSQL, data,
        conflict={"constraint": "uq_energy_cost_accrued_point_start_tariff"}, newer_col="computed_at",
    )


def upsert_energy_run_cost(sess: Session, data: dict[str, Any]) -> bool:
    return _upsert(
        sess, EnergyRunCostSQL, data,
        conflict={"index_elements": ["intent_id"]}, newer_col="computed_at",
    )


def upsert_energy_bill_actual(sess: Session, data: dict[str, Any]) -> bool:
    return _upsert(
        sess, EnergyBillActualSQL, data,
        conflict={"constraint": "uq_energy_bill_actual_period"}, newer_col="retrieved_at",
    )


def upsert_energy_bill_forecast(sess: Session, data: dict[str, Any]) -> bool:
    return _upsert(
        sess, EnergyBillForecastSQL, data,
        conflict={"constraint": "uq_energy_bill_forecast_period_as_of"}, newer_col="retrieved_at",
    )


def upsert_energy_reconcile(sess: Session, data: dict[str, Any]) -> bool:
    return _upsert(
        sess, EnergyReconcileSQL, data,
        conflict={"constraint": "uq_energy_reconcile_key"}, newer_col="computed_at",
    )


def insert_energy_stakes_snapshot(sess: Session, data: dict[str, Any]) -> bool:
    return _insert_once(sess, EnergyStakesSnapshotSQL, data, constraint="uq_energy_stakes_snapshot_as_of")


def insert_energy_importer_status(sess: Session, data: dict[str, Any]) -> bool:
    return _insert_once(sess, EnergyImporterStatusSQL, data, constraint="uq_energy_importer_status_as_of")


ENERGY_UPSERTS: dict[type, Callable[[Session, dict[str, Any]], bool]] = {
    EnergyUsageIntervalSQL: upsert_energy_usage_interval,
    EnergyCostAccruedSQL: upsert_energy_cost_accrued,
    EnergyRunCostSQL: upsert_energy_run_cost,
    EnergyBillActualSQL: upsert_energy_bill_actual,
    EnergyBillForecastSQL: upsert_energy_bill_forecast,
    EnergyReconcileSQL: upsert_energy_reconcile,
    EnergyStakesSnapshotSQL: insert_energy_stakes_snapshot,
    EnergyImporterStatusSQL: insert_energy_importer_status,
}
