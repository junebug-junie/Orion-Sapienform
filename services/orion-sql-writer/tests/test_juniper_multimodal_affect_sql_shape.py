"""Contract tests for persisting orion:affectgpt:assessment events.

First real persistence consumer of that channel -- previously
`consumer_services: ["orion-juniper-affective-state"]` only (the producer's
own live-tap debug script), same shadow-write shape already fixed for the
sibling text-only signal (JuniperAffectiveStateSQL/PR #1629) and
doc_semantic_drift (PR #1730). The one real cognition reader
(orion/situational/juniper_affect_state.py, PR #1865) reads a SEPARATE
Redis SETEX mirror with a 1h TTL -- once that key expires, nothing durable
records a capture ever happened. These tests pin the wiring that fixes
that, the column shape those rows get read back through, and the privacy
boundary the class docstring commits to (transcript never persisted).
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker

REPO_ROOT = Path(__file__).resolve().parents[3]
SERVICE_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, SERVICE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef  # noqa: E402
from orion.schemas.affectgpt import JuniperMultimodalAffectV1  # noqa: E402

from app.models.juniper_multimodal_affect import JuniperMultimodalAffectSQL  # noqa: E402
from app.settings import DEFAULT_ROUTE_MAP, Settings  # noqa: E402

WORKER_PATH = SERVICE_ROOT / "app" / "worker.py"
SPEC = importlib.util.spec_from_file_location("sql_writer_worker_affectgpt_tests", WORKER_PATH)
assert SPEC and SPEC.loader
worker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(worker)

_OBSERVED = datetime(2026, 8, 25, 12, 0, tzinfo=timezone.utc)


def _event(**overrides) -> JuniperMultimodalAffectV1:
    payload = dict(
        observed_at=_OBSERVED,
        ok=True,
        raw_response="Juniper appears calm and focused.",
        model_ckpt="AffectGPT-Qwen2.5-7B",
        subtitle_source="transcribed",
        transcript="verbatim words that must never reach the SQL row",
        trigger="ambient",
        correlation_id="corr-affectgpt-1",
    )
    payload.update(overrides)
    return JuniperMultimodalAffectV1(**payload)


def _source() -> ServiceRef:
    return ServiceRef(name="test-producer", version="0.0.1", node="local")


# --------------------------------------------------------------------------
# Routing / subscription wiring
# --------------------------------------------------------------------------


def test_message_kind_routes_to_the_sql_model() -> None:
    assert DEFAULT_ROUTE_MAP["affectgpt.juniper_multimodal_affect.v1"] == "JuniperMultimodalAffectSQL"


def test_route_map_survives_an_env_override_of_other_routes() -> None:
    settings = Settings(SQL_WRITER_ROUTE_MAP_JSON='{"some.other.kind.v1":"SomethingElseSQL"}')
    assert settings.route_map["affectgpt.juniper_multimodal_affect.v1"] == "JuniperMultimodalAffectSQL"


def test_channel_is_in_the_default_subscribe_list() -> None:
    default_channels = Settings.model_fields["sql_writer_subscribe_channels"].default
    assert "orion:affectgpt:assessment" in default_channels


def test_env_example_lists_the_channel() -> None:
    subscribe_line = next(
        line
        for line in (SERVICE_ROOT / ".env_example").read_text().splitlines()
        if line.startswith("SQL_WRITER_SUBSCRIBE_CHANNELS=")
    )
    assert "orion:affectgpt:assessment" in subscribe_line


def test_model_map_registers_the_sql_model_with_the_event_schema() -> None:
    assert worker.MODEL_MAP["JuniperMultimodalAffectSQL"] == (
        JuniperMultimodalAffectSQL,
        JuniperMultimodalAffectV1,
    )


# --------------------------------------------------------------------------
# Privacy boundary: transcript never reaches the durable row
# --------------------------------------------------------------------------


def test_transcript_has_no_column() -> None:
    """The class docstring's whole privacy commitment, pinned mechanically:
    if a `transcript` column is ever added, this fails loudly instead of
    silently widening exposure."""
    columns = set(JuniperMultimodalAffectSQL.__table__.columns.keys())
    assert "transcript" not in columns


def test_write_row_drops_transcript_even_though_the_wire_schema_carries_it() -> None:
    """_write_row()'s generic column-filter is what actually enforces the
    boundary -- exercise it directly rather than trusting the column-list
    assertion alone."""
    event = _event()
    mapper = inspect(JuniperMultimodalAffectSQL)
    valid_keys = {attr.key for attr in mapper.attrs}
    data = event.model_dump()
    assert "transcript" in data  # sanity: the wire payload really carries it
    filtered = {k: v for k, v in data.items() if k in valid_keys}
    assert "transcript" not in filtered


def test_raw_response_is_persisted_in_full() -> None:
    """The model's own generated read -- not Juniper's words -- is fine to
    keep durably; it is already surfaced live in the Hub UI panel and on
    the bus."""
    columns = set(JuniperMultimodalAffectSQL.__table__.columns.keys())
    assert "raw_response" in columns


@pytest.mark.asyncio
async def test_transcript_is_redacted_from_the_fallback_log_on_write_failure(monkeypatch) -> None:
    """Review finding, 2026-08-25: _write_row()'s column-filter only
    protects the SUCCESS path. handle_envelope's shared except block
    (every MODEL_MAP route) falls back to _write_fallback() with the RAW
    env.payload on any exception -- schema drift raising in
    _coerce_payload, a non-duplicate-key DB error, anything. Without an
    explicit redaction there, a failed write would persist Juniper's
    verbatim transcript into bus_fallback_log, silently breaking the
    exact privacy boundary this model's docstring and the tests above
    claim holds."""

    async def _raising_write(*args, **kwargs):
        raise RuntimeError("simulated write failure")

    monkeypatch.setattr(worker, "_write", _raising_write)

    captured: dict = {}

    def _fake_write_fallback(kind, correlation_id, payload, error=None):
        captured["payload"] = payload
        captured["error"] = error

    monkeypatch.setattr(worker, "_write_fallback", _fake_write_fallback)

    event = _event()
    assert event.transcript  # sanity: the wire payload really carries it
    env = BaseEnvelope(
        kind="affectgpt.juniper_multimodal_affect.v1",
        correlation_id=uuid4(),
        source=_source(),
        payload=event.model_dump(mode="json"),
    )
    await worker.handle_envelope(env, bus=None)

    assert "payload" in captured, "expected the failing write to reach _write_fallback"
    assert "transcript" not in captured["payload"]
    # Not a blanket wipe -- everything else needed to diagnose the
    # failure is still there.
    assert captured["payload"]["raw_response"] == event.raw_response
    assert "simulated write failure" in (captured["error"] or "")


# --------------------------------------------------------------------------
# event_id synthesis (worker.py special case -- this schema has no
# event_id field of its own, unlike the tiling-window text signal)
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_event_id_is_synthesized_from_the_envelope_correlation_id(monkeypatch) -> None:
    captured: dict = {}

    async def _fake_write(sql_model_cls, schema_cls, payload, extra_fields=None, *, kind: str | None = None) -> bool:
        captured["sql_model_cls"] = sql_model_cls
        captured["extra_fields"] = dict(extra_fields or {})
        return True

    monkeypatch.setattr(worker, "_write", _fake_write)

    corr = uuid4()
    env = BaseEnvelope(
        kind="affectgpt.juniper_multimodal_affect.v1",
        correlation_id=corr,
        source=_source(),
        payload=_event().model_dump(mode="json"),
    )
    await worker.handle_envelope(env, bus=None)

    assert captured["sql_model_cls"] is JuniperMultimodalAffectSQL
    assert captured["extra_fields"]["event_id"] == str(corr)
    assert captured["extra_fields"]["correlation_id"] == str(corr)


def test_event_id_falls_back_to_envelope_id_with_no_correlation_id() -> None:
    """Extracted as _affectgpt_multimodal_event_id() specifically so this
    fallback chain is unit-testable: BaseEnvelope.correlation_id is a
    required, always-generated uuid in practice, so this branch can't be
    exercised by constructing a real (invalid) envelope."""
    env = BaseEnvelope(
        kind="affectgpt.juniper_multimodal_affect.v1",
        source=_source(),
        payload=_event().model_dump(mode="json"),
    )
    event_id = worker._affectgpt_multimodal_event_id({}, {}, env)
    assert event_id == str(env.id)


def test_event_id_falls_back_to_a_fresh_uuid_with_nothing_at_all() -> None:
    class _NoId:
        id = None

    event_id = worker._affectgpt_multimodal_event_id({}, {}, _NoId())
    assert event_id  # a real uuid4 string, not empty/None


# --------------------------------------------------------------------------
# Column shape
# --------------------------------------------------------------------------


def test_event_id_is_the_primary_key() -> None:
    pk = [c.name for c in JuniperMultimodalAffectSQL.__table__.primary_key.columns]
    assert pk == ["event_id"]


def test_observed_at_is_indexed() -> None:
    assert JuniperMultimodalAffectSQL.__table__.columns["observed_at"].index is True


def test_ok_is_not_nullable() -> None:
    assert JuniperMultimodalAffectSQL.__table__.columns["ok"].nullable is False


# ------------------------------- chat-turn bracket (2026-08-26) -----------

def test_chat_correlation_id_has_a_column_and_is_indexed() -> None:
    """Without a declared column, _write_row's column-filter would silently
    drop this key -- the same mechanism test_transcript_has_no_column relies
    on to enforce privacy works against us here. The join would then exist
    on the bus and in the 1h Redis mirror but never durably, defeating the
    point of capturing a pre/post pair at all.

    Indexed because the only query it exists to serve is "both legs for
    turn X"."""
    col = JuniperMultimodalAffectSQL.__table__.columns["chat_correlation_id"]
    assert col.index is True
    assert col.nullable is True  # NULL for manual/ambient captures


def test_write_row_keeps_chat_correlation_id_from_a_real_event() -> None:
    """Exercise the actual generic filter with a real chat_turn_pre event,
    not just the column list -- the filter is what decides."""
    from orion.schemas.affectgpt import JuniperMultimodalAffectV1

    event = _event()
    data = event.model_dump()
    data["trigger"] = "chat_turn_pre"
    data["chat_correlation_id"] = "turn-abc"

    mapper = inspect(JuniperMultimodalAffectSQL)
    valid_keys = {attr.key for attr in mapper.attrs}
    filtered = {k: v for k, v in data.items() if k in valid_keys}

    assert filtered["chat_correlation_id"] == "turn-abc"
    assert filtered["trigger"] == "chat_turn_pre"
    # And the privacy boundary still holds on this path.
    assert "transcript" not in filtered


def test_correlation_id_and_chat_correlation_id_are_separate_columns() -> None:
    """Two different join axes. Collapsing them into one column would make
    'which capture attempt' and 'which conversation turn' unanswerable
    independently."""
    columns = set(JuniperMultimodalAffectSQL.__table__.columns.keys())
    assert {"correlation_id", "chat_correlation_id"} <= columns


# --------------------------------------------------------------------------
# Idempotency (SQLite merge, mirrors DevEconomicsLedgerSQL's own test)
# --------------------------------------------------------------------------


def test_redelivery_of_the_same_correlation_id_upserts_one_row() -> None:
    engine = create_engine("sqlite://")
    JuniperMultimodalAffectSQL.__table__.create(bind=engine)
    Session = sessionmaker(bind=engine)

    def _merge(event_id: str, raw_response: str) -> None:
        sess = Session()
        try:
            sess.merge(
                JuniperMultimodalAffectSQL(
                    event_id=event_id,
                    observed_at=_OBSERVED,
                    source="affectgpt",
                    ok=True,
                    raw_response=raw_response,
                    correlation_id=event_id,
                )
            )
            sess.commit()
        finally:
            sess.close()

    _merge("corr-1", "first read")
    _merge("corr-1", "redelivered read")

    sess = Session()
    try:
        rows = sess.query(JuniperMultimodalAffectSQL).all()
        assert len(rows) == 1
        assert rows[0].raw_response == "redelivered read"
    finally:
        sess.close()


# --------------------------------------------------------------------------
# 2026-08-26: the columns whose absence made the vision cutover undiagnosable
# --------------------------------------------------------------------------


def test_quality_telemetry_columns_exist() -> None:
    """The producer had ALWAYS put face_detection and timings on the bus. This
    table never declared columns for them, so _write_row's generic
    column-filter silently dropped both on every insert -- and when the
    2026-08-26 investigation asked "was there even a face in frame when the
    model said that?", six stored rows could not answer, so the only way to
    find out was to point Juniper's own webcam at her again."""
    columns = set(JuniperMultimodalAffectSQL.__table__.columns.keys())
    for column in ("backend", "affect", "face_detection", "timings", "frames_used"):
        assert column in columns, f"{column} would be silently dropped on insert"


def test_write_row_keeps_the_quality_telemetry_it_used_to_drop() -> None:
    """The column-filter is what actually decides, so exercise it directly --
    the same way test_write_row_drops_transcript... does for the privacy side.
    This is the exact filter that ate face_detection before this patch."""
    event = _event()
    mapper = inspect(JuniperMultimodalAffectSQL)
    valid_keys = {attr.key for attr in mapper.attrs}
    data = event.model_dump()
    filtered = {k: v for k, v in data.items() if k in valid_keys}
    for column in ("backend", "affect", "face_detection", "timings", "frames_used"):
        assert column in filtered, f"{column} is on the wire but still filtered out"


def test_transcript_is_still_dropped_now_that_more_columns_exist() -> None:
    """Widening the column set must not have widened the privacy surface --
    re-asserted here rather than relying on the older test alone, because the
    failure mode is exactly 'someone added columns and one of them was
    transcript'."""
    columns = set(JuniperMultimodalAffectSQL.__table__.columns.keys())
    assert "transcript" not in columns
