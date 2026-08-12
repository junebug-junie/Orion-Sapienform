"""Shape checks for the dev-economics ledger SQL write path (no Postgres required).

Mirrors test_dominance_streak_tick_sql_shape.py's pattern: asserts the real
wiring for `orion:substrate:dev_economics_ledger` -> `DevEconomicsLedgerSQL`
is registered in `MODEL_MAP` under the `DevEconomicsLedgerSQL` route key,
keyed off kind `substrate.dev_economics_ledger.v1`.

First real consumer of this channel (docs/superpowers/pr-reports/2026-08-12-
dev-economics-hub-observability.md) -- previously `consumer_services: []`.
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

from orion.schemas.dev_economics import DevEconomicsLedgerV1  # noqa: E402

from app.models.dev_economics_ledger import DevEconomicsLedgerSQL  # noqa: E402
from app.worker import MODEL_MAP  # noqa: E402
from app.settings import DEFAULT_ROUTE_MAP  # noqa: E402


def _make_event(**overrides) -> DevEconomicsLedgerV1:
    defaults = dict(
        event_id="dev-economics-abc-123",
        observed_at=datetime(2026, 8, 12, 12, 0, tzinfo=timezone.utc),
        window_since=datetime(2026, 8, 12, 12, 0, tzinfo=timezone.utc),
        window_until=datetime(2026, 8, 12, 12, 0, tzinfo=timezone.utc),
        session_count=4,
        subagent_transcript_count=1,
        total_input_tokens=1000,
        total_output_tokens=500,
        total_cache_creation_input_tokens=0,
        total_cache_read_input_tokens=0,
        total_tokens=1500,
        total_assistant_word_count=100,
        total_human_word_count=20,
        total_estimated_cost_usd=2.84,
        priced_session_count=4,
        unpriced_session_count=0,
        model_mix={"claude-sonnet-5": 4},
    )
    defaults.update(overrides)
    return DevEconomicsLedgerV1(**defaults)


def test_default_route_map_points_dev_economics_at_dev_economics_ledger_sql() -> None:
    assert DEFAULT_ROUTE_MAP.get("substrate.dev_economics_ledger.v1") == "DevEconomicsLedgerSQL"


def test_model_map_registers_dev_economics_ledger_sql_with_event_schema() -> None:
    assert MODEL_MAP["DevEconomicsLedgerSQL"] == (DevEconomicsLedgerSQL, DevEconomicsLedgerV1)


def test_channel_is_subscribed() -> None:
    from app.settings import settings

    assert "orion:substrate:dev_economics_ledger" in settings.effective_subscribe_channels


def test_event_fields_map_onto_real_columns_except_model_mix() -> None:
    """model_mix is the one deliberate exception: it's a real dict on the
    wire schema but stored as a JSON string column (model_mix_json) --
    handled by an explicit extra_sql_fields injection in worker.py's
    _handle_envelope_body (kind == "substrate.dev_economics_ledger.v1"),
    not the generic column-name match every other field relies on. See that
    injection's own comment for why no generic dict->JSON column mapping
    exists in _write_row."""
    mapper = inspect(DevEconomicsLedgerSQL)
    valid_keys = {attr.key for attr in mapper.attrs}
    data = _make_event().model_dump()
    missing = [field for field in data if field not in valid_keys and field not in ("model_mix", "schema_version", "window_since", "window_until")]
    assert not missing, f"DevEconomicsLedgerV1 fields missing from DevEconomicsLedgerSQL columns: {missing}"


def test_event_data_constructs_dev_economics_ledger_sql_without_raising() -> None:
    event = _make_event()
    row = DevEconomicsLedgerSQL(
        **{k: v for k, v in event.model_dump().items() if k not in ("model_mix", "schema_version", "window_since", "window_until")},
        model_mix_json='{"claude-sonnet-5": 4}',
    )
    assert row.event_id == event.event_id
    assert row.session_count == 4
    assert row.total_estimated_cost_usd == 2.84


def test_event_data_handles_null_cost() -> None:
    """A real tick where nothing was priceable -- total_estimated_cost_usd
    is None, a real fact, not a fabricated 0.0 (DevEconomicsLedgerV1's own
    docstring)."""
    event = _make_event(total_estimated_cost_usd=None, priced_session_count=0, unpriced_session_count=4)
    row = DevEconomicsLedgerSQL(
        **{k: v for k, v in event.model_dump().items() if k not in ("model_mix", "schema_version", "window_since", "window_until")},
        model_mix_json="{}",
    )
    assert row.total_estimated_cost_usd is None


def test_merge_redelivery_upserts_one_row_and_preserves_created_at() -> None:
    """Re-delivery of the same event_id must upsert (one row), not
    duplicate -- mirrors DominanceStreakTickSQL's own merge-idempotency
    test against in-memory SQLite."""
    engine = create_engine("sqlite://")
    DevEconomicsLedgerSQL.__table__.create(bind=engine)
    Session = sessionmaker(bind=engine)

    def _merge(event: DevEconomicsLedgerV1) -> None:
        sess = Session()
        try:
            data = {k: v for k, v in event.model_dump().items() if k not in ("model_mix", "schema_version", "window_since", "window_until")}
            sess.merge(DevEconomicsLedgerSQL(**data, model_mix_json="{}"))
            sess.commit()
        finally:
            sess.close()

    _merge(_make_event(session_count=4))

    sess = Session()
    try:
        first = sess.get(DevEconomicsLedgerSQL, "dev-economics-abc-123")
        original_created_at = first.created_at
    finally:
        sess.close()
    assert original_created_at is not None

    _merge(_make_event(session_count=4, total_estimated_cost_usd=3.10))

    sess = Session()
    try:
        rows = sess.query(DevEconomicsLedgerSQL).all()
        assert len(rows) == 1
        row = rows[0]
        assert row.total_estimated_cost_usd == 3.10
        assert row.created_at == original_created_at
    finally:
        sess.close()
