"""Compile-time shape checks for the chat_stance_belief_log SQL write path
(no Postgres required) -- self-model rebuild arc, 2026-09-05. Real wiring:
`ChatStanceBeliefLogSQL` is registered in `MODEL_MAP` under its own route
key, keyed off kind `chat_stance.belief.write.v1`, and every field on the
real `ChatStanceBeliefLogV1` producer schema maps onto a real column on
`ChatStanceBeliefLogSQL`.

Review finding (2026-09-05): the new route wasn't covered by a real payload/
column test, only by generic route-map completeness -- this closes that gap.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from sqlalchemy import inspect

REPO_ROOT = Path(__file__).resolve().parents[3]
SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT), str(SERVICE_ROOT)]

from orion.schemas.chat_stance_belief import ChatStanceBeliefLogV1  # noqa: E402

from app.models.chat_stance_belief import ChatStanceBeliefLogSQL  # noqa: E402
from app.worker import MODEL_MAP  # noqa: E402
from app.settings import DEFAULT_ROUTE_MAP, Settings  # noqa: E402


def _make_payload(**overrides) -> ChatStanceBeliefLogV1:
    defaults = dict(
        correlation_id="corr-1",
        session_id="sess-1",
        shift_kind="REPAIR",
        anchor_summary="anchors this turn: orion, relationship(degraded)",
        degraded_producers=["producer_x"],
        lineage_summary='{"orion": "producer_x"}',
    )
    defaults.update(overrides)
    return ChatStanceBeliefLogV1(**defaults)


def test_the_channel_is_actually_subscribed() -> None:
    """Review finding: registering a channel/route/model is not the same
    as subscribing to it -- confirmed live on this exact patch, the channel
    was registered everywhere except SQL_WRITER_SUBSCRIBE_CHANNELS, so the
    whole feature was a silent no-op until this was caught. Both the
    .env_example list and a code-default-only Settings instance must carry
    the new channel."""
    example = Path(__file__).resolve().parents[1] / ".env_example"
    raw = next(
        line.split("=", 1)[1].strip()
        for line in example.read_text().splitlines()
        if line.startswith("SQL_WRITER_SUBSCRIBE_CHANNELS=")
    )
    assert "orion:chat_stance:belief:write" in json.loads(raw)

    stale = Settings(SQL_WRITER_SUBSCRIBE_CHANNELS=["orion:biometrics:summary"])
    assert "orion:chat_stance:belief:write" in stale.effective_subscribe_channels


def test_default_route_map_points_chat_stance_belief_write_v1_at_chat_stance_belief_log_sql() -> None:
    assert DEFAULT_ROUTE_MAP.get("chat_stance.belief.write.v1") == "ChatStanceBeliefLogSQL"


def test_model_map_registers_chat_stance_belief_log_sql_with_its_schema() -> None:
    assert MODEL_MAP["ChatStanceBeliefLogSQL"] == (ChatStanceBeliefLogSQL, ChatStanceBeliefLogV1)


def test_chat_stance_belief_log_v1_fields_map_onto_real_columns() -> None:
    mapper = inspect(ChatStanceBeliefLogSQL)
    valid_keys = {attr.key for attr in mapper.attrs}

    payload = _make_payload()
    data = payload.model_dump(mode="json")

    missing = [field for field in data if field not in valid_keys]
    assert not missing, f"ChatStanceBeliefLogV1 fields missing from ChatStanceBeliefLogSQL columns: {missing}"


def test_chat_stance_belief_log_v1_data_constructs_sql_row_without_raising() -> None:
    payload = _make_payload()
    data = payload.model_dump(mode="json")

    row = ChatStanceBeliefLogSQL(**data)

    assert row.entry_id == payload.entry_id
    assert row.correlation_id == payload.correlation_id
    assert row.session_id == payload.session_id
    assert row.shift_kind == payload.shift_kind
    assert row.anchor_summary == payload.anchor_summary
    assert row.degraded_producers == payload.degraded_producers
    assert row.lineage_summary == payload.lineage_summary


def test_chat_stance_belief_log_v1_optional_fields_construct_row_without_raising() -> None:
    payload = _make_payload(correlation_id=None, session_id=None, shift_kind=None, anchor_summary=None, lineage_summary=None, degraded_producers=[])
    data = payload.model_dump(mode="json")

    row = ChatStanceBeliefLogSQL(**data)

    assert row.shift_kind is None
    assert row.degraded_producers == []
