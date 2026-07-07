"""Shape checks for the action-outcome SQL write path (no Postgres required).

Asserts the real wiring for Option C (bus -> sql-writer -> SQL read):
`ActionOutcomeSQL` is registered in `MODEL_MAP` under the `ActionOutcomeSQL`
route key, keyed off kind `action.outcome.emit.v1`, and every field on the
producer schema `ActionOutcomeEmitV1` maps onto a real column.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import inspect

REPO_ROOT = Path(__file__).resolve().parents[3]
SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT), str(SERVICE_ROOT)]

from orion.autonomy.models import ActionOutcomeEmitV1, ActionOutcomeRefV1  # noqa: E402

from app.models.action_outcome import ActionOutcomeSQL  # noqa: E402
from app.worker import MODEL_MAP  # noqa: E402
from app.settings import DEFAULT_ROUTE_MAP  # noqa: E402


def _make_emit(**overrides) -> ActionOutcomeEmitV1:
    defaults = dict(
        subject="orion",
        action_id="fetch-abc-123",
        kind="web.fetch.readonly",
        summary="fetched 2 article(s)",
        success=True,
        surprise=0.0,
        observed_at=datetime(2026, 7, 6, 12, 0, tzinfo=timezone.utc),
    )
    defaults.update(overrides)
    return ActionOutcomeEmitV1(**defaults)


def test_default_route_map_points_action_outcome_at_action_outcome_sql() -> None:
    assert DEFAULT_ROUTE_MAP.get("action.outcome.emit.v1") == "ActionOutcomeSQL"


def test_model_map_registers_action_outcome_sql_with_emit_schema() -> None:
    assert MODEL_MAP["ActionOutcomeSQL"] == (ActionOutcomeSQL, ActionOutcomeEmitV1)


def test_channel_is_subscribed() -> None:
    from app.settings import settings

    assert "orion:autonomy:action:outcome" in settings.effective_subscribe_channels


def test_emit_fields_map_onto_real_columns() -> None:
    mapper = inspect(ActionOutcomeSQL)
    valid_keys = {attr.key for attr in mapper.attrs}
    data = _make_emit().model_dump()
    missing = [field for field in data if field not in valid_keys]
    assert not missing, f"ActionOutcomeEmitV1 fields missing from ActionOutcomeSQL columns: {missing}"


def test_emit_data_constructs_action_outcome_sql_without_raising() -> None:
    emit = _make_emit()
    row = ActionOutcomeSQL(**emit.model_dump())
    assert row.action_id == emit.action_id
    assert row.subject == emit.subject
    assert row.success is True
    assert row.surprise == 0.0


def test_emit_roundtrips_from_outcome() -> None:
    ref = ActionOutcomeRefV1(
        action_id="fetch-1",
        kind="web.fetch.readonly",
        summary="fetched 1 article(s)",
        success=True,
        surprise=0.2,
        observed_at=datetime(2026, 7, 6, 12, 0, tzinfo=timezone.utc),
    )
    emit = ActionOutcomeEmitV1.from_outcome(subject="orion", outcome=ref)
    assert emit.subject == "orion"
    back = emit.to_outcome()
    assert back == ref
