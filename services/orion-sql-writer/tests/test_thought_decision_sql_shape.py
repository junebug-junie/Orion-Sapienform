"""Shape checks for the ThoughtDecisionSQL write path (no Postgres required).

Asserts the real wiring for bus -> sql-writer -> SQL read: ThoughtEventV1
already broadcasts live on orion:thought:artifact (producer: orion-thought,
consumer: orion-equilibrium-service's metacog gate) but was never durably
persisted anywhere queryable by correlation_id. This adds sql-writer as a
second consumer of the *same, already-live* channel rather than a new
publish path -- ThoughtDecisionSQL under its own route key, keyed off kind
`thought.event.v1` (both in DEFAULT_ROUTE_MAP and the operator-facing
.env_example), and confirms the channel is actually subscribed to.

The redaction test that matters is `test_real_write_row_persists_stance_content_and_drops_the_rest`:
it calls the REAL `worker._write_row()` (monkeypatching only its DB session,
not its filtering logic) against a live sqlite table, so it exercises the
actual column-name -> attribute-key mapping and generic filter production
code uses -- not a local reimplementation of it. The other `_row_dict()`-based
tests below are a cheaper, non-DB shape check on top of that; they mirror
`_write_row()`'s filter but do not call it directly, so they alone would not
catch a regression in the real filtering/column-mapping mechanism -- only in
which columns ThoughtDecisionSQL declares.

``imperative``/``tone``/``strain_refs``/``stance_harness_slice`` are
persisted unredacted as of the Turn Trace full-implementation pass (this is
the operator debug surface, an explicit choice -- see
orion/schemas/thought.py's ThoughtDecisionRecordV1 docstring).
``evidence_refs``/``grounding_capsule``/``autonomy_slice`` still never reach
this table -- those carry denser identity/relationship/memory-digest content
than a stance decision needs to be inspectable.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import scoped_session, sessionmaker

REPO_ROOT = Path(__file__).resolve().parents[3]
SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT), str(SERVICE_ROOT)]

from orion.schemas.thought import StanceHarnessSliceV1, ThoughtEventV1  # noqa: E402

from app.models.thought_decision import ThoughtDecisionSQL  # noqa: E402
from app import worker  # noqa: E402
from app.worker import MODEL_MAP  # noqa: E402
from app.settings import DEFAULT_ROUTE_MAP, Settings  # noqa: E402

CHANNEL = "orion:thought:artifact"


def _thought(**overrides) -> ThoughtEventV1:
    defaults = dict(
        event_id="thought-1",
        correlation_id="corr-1",
        session_id="sess-1",
        created_at=datetime.now(timezone.utc),
        imperative="Answer directly and stay grounded in what was said.",
        tone="steady",
        strain_refs=["strain:continuity"],
        evidence_refs=["evidence:1"],
        repair_pressure_level=0.42,
        trust_rupture_score=0.1,
        disposition="proceed",
        disposition_reasons=["no boundary concerns"],
        boundary_register=False,
        stance_harness_slice=StanceHarnessSliceV1(
            task_mode="chat",
            conversation_frame="direct",
            answer_strategy="concise",
        ),
        llm_profile="brain",
        producer="stance_react_v1",
        model_id="MODEL_SONNET",
    )
    defaults.update(overrides)
    return ThoughtEventV1(**defaults)


def _row_dict(thought: ThoughtEventV1) -> dict:
    """Mirror _write()'s obj.model_dump() (python mode, native datetimes --
    matches the real path exactly, not the JSON-string mode)."""
    data = thought.model_dump()
    mapper = inspect(ThoughtDecisionSQL)
    valid_keys = {attr.key for attr in mapper.attrs}
    row = {k: v for k, v in data.items() if k in valid_keys}
    return row


@pytest.fixture()
def sqlite_worker_session(monkeypatch):
    """Redirect worker.get_session()/remove_session() to a real sqlite-backed
    session, WITHOUT touching _write_row's own filtering/coercion logic --
    proves the actual production write path, not a reimplementation of it.
    """
    engine = create_engine("sqlite://")
    ThoughtDecisionSQL.__table__.create(bind=engine)
    session_factory = scoped_session(sessionmaker(bind=engine))
    monkeypatch.setattr(worker, "get_session", session_factory)
    monkeypatch.setattr(worker, "remove_session", session_factory.remove)
    try:
        yield engine, session_factory
    finally:
        session_factory.remove()


def test_real_write_row_persists_stance_content_and_drops_the_rest(sqlite_worker_session) -> None:
    """The load-bearing redaction test: calls the real worker._write_row()."""
    _engine, session_factory = sqlite_worker_session
    thought = _thought()
    data = thought.model_dump()

    ok = worker._write_row(ThoughtDecisionSQL, data)
    assert ok is True

    sess = session_factory()
    # ThoughtEventV1 has no top-level "id" field (only event_id) and
    # _write_row has no special-cased id-from-event_id fill for this model
    # (unlike e.g. ChatMessageSQL) -- id is left to its column-level
    # uuid4 default, so the real row is only findable by event_id, not by
    # primary-key lookup.
    fetched = sess.query(ThoughtDecisionSQL).filter_by(event_id=thought.event_id).first()
    assert fetched is not None
    assert fetched.disposition == "proceed"
    assert fetched.correlation_id == "corr-1"
    assert fetched.imperative == thought.imperative
    assert fetched.tone == thought.tone
    assert fetched.strain_refs == thought.strain_refs
    assert fetched.stance_harness_slice == thought.stance_harness_slice.model_dump()
    for sensitive_key in (
        "evidence_refs",
        "grounding_capsule",
        "autonomy_slice",
    ):
        assert not hasattr(fetched, sensitive_key), f"{sensitive_key} leaked onto the real persisted row"


def test_default_route_map_points_thought_event_at_thought_decision_sql() -> None:
    assert DEFAULT_ROUTE_MAP.get("thought.event.v1") == "ThoughtDecisionSQL"


def test_model_map_registers_thought_decision_sql_with_thought_event_v1_schema() -> None:
    assert MODEL_MAP["ThoughtDecisionSQL"] == (ThoughtDecisionSQL, ThoughtEventV1)


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


def test_env_example_route_map_json_includes_thought_decision() -> None:
    env_example = (SERVICE_ROOT / ".env_example").read_text()
    for line in env_example.splitlines():
        if line.startswith("SQL_WRITER_ROUTE_MAP_JSON="):
            assert '"thought.event.v1":"ThoughtDecisionSQL"' in line
            return
    raise AssertionError("SQL_WRITER_ROUTE_MAP_JSON not found in .env_example")


def test_row_dict_persists_stance_content_and_drops_the_rest() -> None:
    """imperative/tone/strain_refs/stance_harness_slice reach the row; the
    denser identity/relationship fields still don't."""
    thought = _thought()
    row = _row_dict(thought)
    for kept_key in ("imperative", "tone", "strain_refs", "stance_harness_slice"):
        assert kept_key in row, f"{kept_key} unexpectedly dropped from ThoughtDecisionSQL row"
    for sensitive_key in (
        "evidence_refs",
        "grounding_capsule",
        "autonomy_slice",
    ):
        assert sensitive_key not in row, f"{sensitive_key} leaked into ThoughtDecisionSQL row"


def test_row_dict_constructs_thought_decision_sql_without_raising() -> None:
    thought = _thought()
    row = ThoughtDecisionSQL(**_row_dict(thought))
    assert row.disposition == "proceed"
    assert row.correlation_id == "corr-1"
    assert row.disposition_reasons == ["no boundary concerns"]


def test_write_and_read_roundtrip_sqlite() -> None:
    """ORM-construction path (not _write_row) -- id is left to its column
    default, same as production, and looked up by correlation_id rather than
    primary key (ThoughtDecisionSQL.id is an opaque uuid4, not event_id)."""
    engine = create_engine("sqlite://")
    ThoughtDecisionSQL.__table__.create(bind=engine)
    Session = sessionmaker(bind=engine)

    thought = _thought(
        disposition="defer",
        disposition_reasons=["needs more context"],
        boundary_register=True,
    )
    row = ThoughtDecisionSQL(**_row_dict(thought))

    sess = Session()
    try:
        sess.add(row)
        sess.commit()
        fetched = sess.query(ThoughtDecisionSQL).filter_by(correlation_id="corr-1").first()
        assert fetched is not None
        assert fetched.disposition == "defer"
        assert fetched.disposition_reasons == ["needs more context"]
        assert fetched.boundary_register is True
        assert fetched.correlation_id == "corr-1"
    finally:
        sess.close()
