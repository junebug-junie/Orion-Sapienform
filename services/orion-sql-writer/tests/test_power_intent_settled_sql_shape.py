"""Column shape and routing for power.intent.settled.v1 -> power_intent_settled.

Written with the two blockers from the orion_biometrics_cluster review already in mind:
`_write_row` drops an unmapped key silently, and SQL_WRITER_SUBSCRIBE_CHANNELS replaces
rather than merges, so a correct route with no subscription is a route to nowhere.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from sqlalchemy import inspect as sa_inspect

from app.models.power_intent_settled import PowerIntentSettledSQL
from app.settings import Settings

from orion.schemas.power import PowerIntentSettledV1


def _columns() -> dict:
    return {c.key: c for c in sa_inspect(PowerIntentSettledSQL).columns}


def test_every_payload_field_has_a_column() -> None:
    """No silent drops -- _write_row filters against the mapper and discards the rest
    without raising."""
    cols = _columns()
    for field in PowerIntentSettledV1.model_fields:
        assert field in cols, f"payload field {field!r} has no column"


def test_the_channel_is_actually_subscribed() -> None:
    """The blocker that made the sibling table inert: the code default does not survive
    an env override of this field, because it replaces rather than merges."""
    example = Path(__file__).resolve().parents[1] / ".env_example"
    raw = next(
        line.split("=", 1)[1].strip()
        for line in example.read_text().splitlines()
        if line.startswith("SQL_WRITER_SUBSCRIBE_CHANNELS=")
    )
    assert "orion:power:intent:settled" in json.loads(raw)

    stale = Settings(SQL_WRITER_SUBSCRIBE_CHANNELS=["orion:biometrics:summary"])
    assert "orion:power:intent:settled" in stale.effective_subscribe_channels


def test_the_kind_routes_to_this_table() -> None:
    from app.worker import MODEL_MAP

    model_cls, schema_cls = MODEL_MAP["PowerIntentSettledSQL"]
    assert model_cls.__tablename__ == "power_intent_settled"
    assert schema_cls.__name__ == "PowerIntentSettledV1"
    assert Settings().route_map.get("power.intent.settled.v1") == "PowerIntentSettledSQL"


def test_unknown_columns_are_nullable_so_zero_always_means_measured() -> None:
    """A zero in any of these is a MEASUREMENT. Writing one where we mean 'not known'
    is how an unmeasured workload comes to look perfectly predicted."""
    cols = _columns()
    for name in (
        "expected_watts",
        "residual_watts",
        "actual_peak_watts",
        "actual_mean_watts",
        "energy_joules",
        "baseline_watts",
        "achieved_sample_hz",
    ):
        assert cols[name].nullable, f"{name} must be nullable -- null means UNKNOWN"


def test_outcome_is_not_nullable_because_it_is_how_unknown_is_expressed() -> None:
    """Every actual_* being null is ambiguous on its own; `outcome` is what
    disambiguates 'we did not see' from 'we saw nothing drawn'. A row without it cannot
    be read correctly, so it must never be absent."""
    assert _columns()["outcome"].nullable is False


def test_one_settlement_per_intent() -> None:
    """This table is read as a distribution. A redelivered envelope adding a second row
    silently reweights it."""
    uniques = {
        tuple(sorted(c.name for c in con.columns))
        for con in PowerIntentSettledSQL.__table__.constraints
        if con.__class__.__name__ == "UniqueConstraint"
    }
    assert ("intent_id",) in uniques


def test_a_real_payload_round_trips_through_the_column_filter() -> None:
    """Drives the same filtering _write_row does, against a validated payload."""
    now = datetime(2026, 8, 28, 5, 0, tzinfo=timezone.utc)
    payload = PowerIntentSettledV1(
        intent_id="i1",
        workload_kind="reverie_diffusion",
        node="circe",
        gpu_index=2,
        outcome="settled",
        window_start=now,
        window_end=now + timedelta(seconds=8),
        sample_count=8,
        achieved_sample_hz=1.0,
        actual_peak_watts=220.0,
        actual_mean_watts=140.0,
        baseline_watts=42.0,
        energy_joules=1120.0,
    ).model_dump()

    cols = set(_columns())
    kept = {k: v for k, v in payload.items() if k in cols}
    assert kept["outcome"] == "settled"
    assert kept["actual_peak_watts"] == 220.0
    assert kept["expected_watts"] is None
    assert kept["residual_watts"] is None
    assert set(payload) - cols == set(), "payload fields would be silently dropped"
