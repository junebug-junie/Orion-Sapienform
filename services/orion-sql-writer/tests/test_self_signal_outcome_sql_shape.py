"""Compile-time shape checks for the self-signal outcome association write path (no Postgres):
subscribed even under a stale operator channel list, routed, every schema field lands on a
column, hour rows upsert while association rows are append-only, and real producer rows map."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import inspect

REPO_ROOT = Path(__file__).resolve().parents[3]
SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT), str(SERVICE_ROOT)]

from orion.schemas.registry import resolve  # noqa: E402
from orion.schemas.self_signal_outcome import (  # noqa: E402
    SELF_SIGNAL_ASSOCIATION_CHANNEL,
    SELF_SIGNAL_ASSOCIATION_KIND,
    SELF_SIGNAL_HOUR_CHANNEL,
    SELF_SIGNAL_HOUR_KIND,
    OutcomeHourV1,
    SelfSignalHourV1,
    SelfSignalOutcomeAssociationV1,
    SignalHourV1,
)

from app.models.self_signal_outcome import SelfSignalHourSQL, SelfSignalOutcomeAssociationSQL  # noqa: E402
from app.settings import DEFAULT_ROUTE_MAP, Settings  # noqa: E402
from app.worker import INSERT_ONLY_MODELS, MODEL_MAP  # noqa: E402

PAIRS = (
    (SELF_SIGNAL_HOUR_CHANNEL, SELF_SIGNAL_HOUR_KIND, "SelfSignalHourSQL", SelfSignalHourSQL, SelfSignalHourV1),
    (SELF_SIGNAL_ASSOCIATION_CHANNEL, SELF_SIGNAL_ASSOCIATION_KIND, "SelfSignalOutcomeAssociationSQL",
     SelfSignalOutcomeAssociationSQL, SelfSignalOutcomeAssociationV1),
)


def _env_example(key: str):
    raw = next(line.split("=", 1)[1].strip() for line in (SERVICE_ROOT / ".env_example").read_text().splitlines()
               if line.startswith(f"{key}="))
    return json.loads(raw)


def test_channels_are_actually_subscribed() -> None:
    stale = Settings(SQL_WRITER_SUBSCRIBE_CHANNELS=["orion:biometrics:summary"])
    for channel, *_ in PAIRS:
        assert channel in _env_example("SQL_WRITER_SUBSCRIBE_CHANNELS")
        assert channel in stale.effective_subscribe_channels


def test_route_map_and_model_map_agree() -> None:
    for _, kind, name, sql, schema in PAIRS:
        assert DEFAULT_ROUTE_MAP[kind] == name
        assert _env_example("SQL_WRITER_ROUTE_MAP_JSON")[kind] == name
        assert MODEL_MAP[name] == (sql, schema)
        assert resolve(schema.__name__) is schema
    # An hour row is rewritten when a failed publish is retried; associations are a history.
    assert SelfSignalHourSQL not in INSERT_ONLY_MODELS
    assert SelfSignalOutcomeAssociationSQL in INSERT_ONLY_MODELS


def test_every_schema_field_lands_on_a_column() -> None:
    for *_, sql, schema in PAIRS:
        columns = {c.key for c in inspect(sql).columns}
        for name in schema.model_fields:
            assert name in columns, (sql.__tablename__, name)


def test_producer_rows_construct_the_sql_models() -> None:
    h = datetime(2026, 10, 14, 3, tzinfo=timezone.utc)
    hour = SelfSignalHourV1(
        hour_id="ssh:2026-10-14T03:00:00Z", hour_start=h, computed_at=h,
        signals=[SignalHourV1(signal_id="node:substrate.biometrics", condition="high", n_readings=100,
                              n_judged=1, n_hot=12, judged_sec=3600.0, hot_sec=420.0)],
        outcomes=[OutcomeHourV1(outcome="turn_failed", exposure=4, events=1, source="x")],
        history_hours=168.0,
    )
    obj = SelfSignalHourSQL(**hour.model_dump(mode="json"))
    assert obj.signals[0]["condition"] == "high" and obj.outcomes[0]["events"] == 1
    assoc = SelfSignalOutcomeAssociationV1(
        association_id="ssa-0123456789", computed_at=h, signal_id="node:substrate.biometrics",
        outcome="turn_failed", n_high=30, events_high=18, n_normal=60, events_normal=6, verdict="tends_more",
        clause="c",
    )
    assert SelfSignalOutcomeAssociationSQL(**assoc.model_dump()).verdict == "tends_more"


def test_association_rows_have_bounded_retention() -> None:
    """Readers need only each pair's last few hourly rows; the table no longer grows forever."""
    from app.grammar_retention_loop import retention_days_for
    from app.grammar_truth import GRAMMAR_RETENTION_TABLES, _EXTRA_RETENTION_TABLES, _other_retention_truth_blocks

    settings = Settings(SQL_WRITER_SUBSCRIBE_CHANNELS=[])
    assert settings.self_signal_outcome_association_retention_days == 30
    assert retention_days_for(settings)["self_signal_outcome_association"] == 30
    assert "self_signal_outcome_association" in dict(GRAMMAR_RETENTION_TABLES)
    assert "self_signal_outcome_association" in _EXTRA_RETENTION_TABLES
    assert "self_signal_outcome_association" in _other_retention_truth_blocks(settings)
    assert "SELF_SIGNAL_OUTCOME_ASSOCIATION_RETENTION_DAYS=30" in (SERVICE_ROOT / ".env_example").read_text()
    assert "SELF_SIGNAL_OUTCOME_ASSOCIATION_RETENTION_DAYS" in (SERVICE_ROOT / "docker-compose.yml").read_text()
    # Every managed table keeps a >= 5 s share of the cycle budget.
    assert settings.grammar_retention_periodic_max_cycle_sec / len(GRAMMAR_RETENTION_TABLES) >= 5.0
    # The prune ages rows by created_at, which this table has.
    assert "created_at" in inspect(SelfSignalOutcomeAssociationSQL).columns
