"""Movement III — reverie makes a checkable prediction about the room, then
finds out if it was right (docs/superpowers/specs/2026-08-12-perception-
frontier-design.md).

Mirrors the split ac633a411 (P5 perception context) used: schema round-trip +
prompt rendering as pure unit tests, store functions mocked at the sqlalchemy
engine boundary (no real DB required), and end-to-end coverage of the actual
settings-gated scoring branch inside run_reverie_once so a future refactor
that drops the flag check / asyncio.to_thread wiring / judge-call plumbing
fails a test, not just silently no-ops the feature.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from jinja2 import Environment

from orion.schemas.attention_frame import AttentionBroadcastProjectionV1, AttentionFrameV1, OpenLoopV1
from orion.schemas.reverie import SpontaneousThoughtV1
from orion.schemas.thought import CoalitionSnapshotV1

GROUNDED_TEXT = (
    "I keep returning to the unresolved thread ol-1 — the conflict about the "
    "deployment plan has not discharged and it is pulling attention."
)

_NARRATE_PROMPT = (
    Path(__file__).resolve().parents[3] / "orion" / "cognition" / "prompts" / "reverie_narrate.j2"
)
_JUDGE_PROMPT = (
    Path(__file__).resolve().parents[3]
    / "orion"
    / "cognition"
    / "prompts"
    / "reverie_expectation_judge.j2"
)


def _render(path: Path, **ctx) -> str:
    template = Environment().from_string(path.read_text())
    return template.render(**ctx)


def _coalition(attended=("n-1",), open_loops=("ol-1",), selected="ol-1"):
    return CoalitionSnapshotV1(
        attended_node_ids=list(attended),
        selected_open_loop_id=selected,
        open_loop_ids=list(open_loops),
        generated_at="2026-07-06T00:00:00Z",
    )


def _broadcast(attended=("n-1",), loops=(("ol-1", {"salience": 0.8}),), selected="ol-1"):
    frame = AttentionFrameV1(
        open_loops=[OpenLoopV1(id=oid, description="d", **scores) for oid, scores in loops],
    )
    return AttentionBroadcastProjectionV1(
        frame=frame,
        attended_node_ids=list(attended),
        selected_open_loop_id=selected,
        coalition_stability_score=0.4,
    )


# --- 1. schema round-trip ------------------------------------------------------


def test_schema_round_trip_with_expectation_fields():
    checkable = datetime(2026, 8, 19, 12, 0, 0, tzinfo=timezone.utc)
    scored = checkable + timedelta(minutes=5)
    t = SpontaneousThoughtV1(
        thought_id="t1",
        correlation_id="c1",
        coalition=_coalition(),
        interpretation=GROUNDED_TEXT,
        evidence_refs=["n-1"],
        expectation="the door will still be closed",
        expectation_checkable_by=checkable,
        expectation_verdict="confirmed",
        expectation_scored_at=scored,
    )
    dumped = t.model_dump(mode="json")
    restored = SpontaneousThoughtV1.model_validate(dumped)
    assert restored.expectation == "the door will still be closed"
    assert restored.expectation_verdict == "confirmed"
    assert restored.expectation_checkable_by == checkable
    assert restored.expectation_scored_at == scored


def test_schema_expectation_fields_default_none():
    t = SpontaneousThoughtV1(
        thought_id="t1", correlation_id="c1", coalition=_coalition(),
        interpretation=GROUNDED_TEXT, evidence_refs=["n-1"],
    )
    assert t.expectation is None
    assert t.expectation_checkable_by is None
    assert t.expectation_verdict is None
    assert t.expectation_scored_at is None


def test_schema_rejects_expectation_over_max_length():
    with pytest.raises(Exception):
        SpontaneousThoughtV1(
            thought_id="t1", correlation_id="c1", coalition=_coalition(),
            interpretation=GROUNDED_TEXT, evidence_refs=["n-1"],
            expectation="x" * 201,
        )


def test_schema_invalid_verdict_literal_rejected():
    with pytest.raises(Exception):
        SpontaneousThoughtV1(
            thought_id="t1", correlation_id="c1", coalition=_coalition(),
            interpretation=GROUNDED_TEXT, evidence_refs=["n-1"],
            expectation_verdict="maybe",
        )


def test_expectation_does_not_widen_grounding_or_hollow_guard():
    """Non-goal: anti-hollow guard stays coalition-only, unchanged."""
    t = SpontaneousThoughtV1(
        thought_id="t1", correlation_id="c1", coalition=None,
        interpretation=GROUNDED_TEXT, evidence_refs=[],
        expectation="the door will still be closed",
    ).marked_hollow()
    assert t.is_hollow() and t.hollow_reason == "absent_coalition"


# --- 2. parse_reverie_payload extracts + truncates expectation ----------------


def test_parse_reverie_payload_extracts_expectation():
    from app import reverie

    raw = {"interpretation": GROUNDED_TEXT, "evidence_refs": ["n-1"], "expectation": "the light stays on"}
    thought = reverie.parse_reverie_payload(
        raw, coalition=_coalition(), correlation_id="c1", broadcast=_broadcast()
    )
    assert thought.expectation == "the light stays on"


def test_parse_reverie_payload_truncates_long_expectation():
    from app import reverie

    raw = {"interpretation": GROUNDED_TEXT, "evidence_refs": ["n-1"], "expectation": "x" * 500}
    thought = reverie.parse_reverie_payload(
        raw, coalition=_coalition(), correlation_id="c1", broadcast=_broadcast()
    )
    assert thought.expectation is not None and len(thought.expectation) == 200


def test_parse_reverie_payload_expectation_absent_by_default():
    from app import reverie

    raw = {"interpretation": GROUNDED_TEXT, "evidence_refs": ["n-1"]}
    thought = reverie.parse_reverie_payload(
        raw, coalition=_coalition(), correlation_id="c1", broadcast=_broadcast()
    )
    assert thought.expectation is None


def test_parse_reverie_payload_blank_expectation_becomes_none():
    from app import reverie

    raw = {"interpretation": GROUNDED_TEXT, "evidence_refs": ["n-1"], "expectation": "   "}
    thought = reverie.parse_reverie_payload(
        raw, coalition=_coalition(), correlation_id="c1", broadcast=_broadcast()
    )
    assert thought.expectation is None


# --- 3. prompt rendering --------------------------------------------------------


def test_narrate_prompt_omits_expectation_instruction_without_percepts():
    from app import reverie

    ctx = reverie.build_reverie_context(_broadcast())
    out = _render(_NARRATE_PROMPT, **ctx)
    assert "falsifiable expectation" not in out
    assert "- expectation:" not in out


def test_narrate_prompt_offers_expectation_field_with_percepts_plain_mode():
    from app import reverie

    percepts = [{"narrative": "a person walks into frame near the door", "age_sec": 12.0}]
    ctx = reverie.build_reverie_context(_broadcast(), recent_percepts=percepts)
    out = _render(_NARRATE_PROMPT, **ctx)
    assert "falsifiable expectation" in out
    assert "- expectation:" in out
    assert "Never force one every tick" in out


def test_narrate_prompt_offers_expectation_field_with_percepts_concern_cards_mode():
    from app import reverie
    from orion.schemas.reverie import ConcernCardV1

    card = ConcernCardV1(
        card_id="c1", coalition_ref="ol-1", human_text="You said the deploy might slip.",
        thread_label="deploy",
    )
    percepts = [{"narrative": "the kitchen light is on", "age_sec": 40.0}]
    ctx = reverie.build_reverie_context(_broadcast(), concern_cards=[card], recent_percepts=percepts)
    out = _render(_NARRATE_PROMPT, **ctx)
    assert "falsifiable expectation" in out
    assert "- expectation:" in out


def test_judge_prompt_renders_expectation_and_percept():
    out = _render(
        _JUDGE_PROMPT,
        expectation="the door will still be closed",
        percept_narrative="the door is closed and no one is present",
    )
    assert "the door will still be closed" in out
    assert "the door is closed and no one is present" in out
    assert '"confirmed"' in out and '"disconfirmed"' in out and '"unscored"' in out


# --- 4. parse_expectation_judge_verdict: fail-closed ----------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ({"verdict": "confirmed"}, "confirmed"),
        ({"verdict": "disconfirmed"}, "disconfirmed"),
        ({"verdict": "unscored"}, "unscored"),
        ({"verdict": "CONFIRMED"}, "confirmed"),
        (json.dumps({"verdict": "disconfirmed"}), "disconfirmed"),
    ],
)
def test_parse_expectation_judge_verdict_recognized(raw, expected):
    from app import reverie

    assert reverie.parse_expectation_judge_verdict(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        {"verdict": "maybe"},
        {},
        {"verdict": ""},
        "not json at all {{{",
        "[]",
        123,
    ],
)
def test_parse_expectation_judge_verdict_fails_closed_to_unscored(raw):
    from app import reverie

    assert reverie.parse_expectation_judge_verdict(raw) == "unscored"


# --- 5. store: load_pending_expectations ----------------------------------------


def _mock_engine(rows):
    engine = MagicMock()
    conn = MagicMock()
    result = MagicMock()
    result.mappings.return_value.all.return_value = rows
    conn.execute.return_value = result
    engine.connect.return_value.__enter__.return_value = conn
    return engine, conn


def test_load_pending_expectations_returns_rows():
    from app import store

    store._expectation_read_engine = None
    row = {
        "thought_id": "t1",
        "expectation": "the door will still be closed",
        "expectation_checkable_by": datetime.now(timezone.utc) - timedelta(minutes=5),
    }
    engine, _ = _mock_engine([row])
    with patch.object(store, "_get_expectation_read_engine", return_value=engine):
        out = store.load_pending_expectations(limit=1)
    assert out == [row]


def test_load_pending_expectations_query_predicates_before_limit():
    from app import store

    store._expectation_read_engine = None
    engine, conn = _mock_engine([])
    with patch.object(store, "_get_expectation_read_engine", return_value=engine):
        store.load_pending_expectations(limit=1)
    query_text = str(conn.execute.call_args[0][0])
    params = conn.execute.call_args[0][1]
    assert "expectation IS NOT NULL" in query_text
    assert "expectation_verdict IS NULL" in query_text
    assert "expectation_checkable_by <= now()" in query_text
    assert query_text.index("expectation IS NOT NULL") < query_text.index("LIMIT")
    assert params == {"limit": 1}


def test_load_pending_expectations_bounded_default_limit_one():
    from app import store

    store._expectation_read_engine = None
    engine, conn = _mock_engine([])
    with patch.object(store, "_get_expectation_read_engine", return_value=engine):
        store.load_pending_expectations()
    params = conn.execute.call_args[0][1]
    assert params["limit"] == 1


def test_load_pending_expectations_limit_zero_short_circuits():
    from app import store

    store._expectation_read_engine = None
    get_engine = MagicMock()
    with patch.object(store, "_get_expectation_read_engine", get_engine):
        out = store.load_pending_expectations(limit=0)
    assert out == []
    get_engine.assert_not_called()


def test_load_pending_expectations_fails_open_on_db_error():
    from app import store

    store._expectation_read_engine = None
    with patch.object(store, "_get_expectation_read_engine", side_effect=RuntimeError("db down")):
        out = store.load_pending_expectations(limit=1)
    assert out == []


def test_expectation_read_engine_has_statement_timeout():
    from app import store

    store._expectation_read_engine = None
    store._expectation_read_engine_url = None
    created = {}

    def _fake_create_engine(url, **kwargs):
        created.update(kwargs)
        return MagicMock()

    with patch("sqlalchemy.create_engine", side_effect=_fake_create_engine):
        store._get_expectation_read_engine()
    assert created.get("connect_args") == {
        "options": f"-c statement_timeout={store._EXPECTATION_QUERY_STATEMENT_TIMEOUT_MS}"
    }
    assert store._EXPECTATION_QUERY_STATEMENT_TIMEOUT_MS == 1500
    store._expectation_read_engine = None
    store._expectation_read_engine_url = None


# --- 6. store: persist_expectation_verdict --------------------------------------


def test_persist_expectation_verdict_writes_update():
    from app import store

    conn = MagicMock()
    engine = MagicMock()
    engine.begin.return_value.__enter__.return_value = conn
    now = datetime.now(timezone.utc)
    with patch.object(store, "_get_engine", return_value=engine):
        store.persist_expectation_verdict("t1", "confirmed", now)
    query_text = str(conn.execute.call_args[0][0])
    params = conn.execute.call_args[0][1]
    assert "UPDATE substrate_reverie_thought" in query_text
    assert "expectation_verdict" in query_text
    assert "expectation_scored_at" in query_text
    assert params == {"thought_id": "t1", "verdict": "confirmed", "scored_at": now}


def test_persist_expectation_verdict_fails_open_never_raises():
    from app import store

    with patch.object(store, "_get_engine", side_effect=RuntimeError("db down")):
        store.persist_expectation_verdict("t1", "confirmed", datetime.now(timezone.utc))  # no raise


def test_persist_expectation_verdict_noop_on_empty_thought_id():
    from app import store

    get_engine = MagicMock()
    with patch.object(store, "_get_engine", get_engine):
        store.persist_expectation_verdict("", "confirmed", datetime.now(timezone.utc))
    get_engine.assert_not_called()


# --- 7. reverie.py scoring branch (end-to-end) ----------------------------------


@pytest.mark.asyncio
async def test_scoring_noop_when_flag_off(monkeypatch):
    """Flag off -> scoring code never runs at all: the pending-load function
    must not even be called."""
    from app import reverie
    from app.settings import settings

    monkeypatch.setattr(settings, "reverie_expectation_scoring_enabled", False)

    def boom(*_a, **_kw):
        raise AssertionError("load_pending_expectations must not be called when disabled")

    monkeypatch.setattr(reverie, "load_pending_expectations", boom)
    bus = AsyncMock()
    await reverie._maybe_score_pending_expectation(bus)


@pytest.mark.asyncio
async def test_scoring_noop_when_no_pending(monkeypatch):
    from app import reverie
    from app.settings import settings

    monkeypatch.setattr(settings, "reverie_expectation_scoring_enabled", True)
    monkeypatch.setattr(reverie, "load_pending_expectations", lambda limit: [])
    persisted = []
    monkeypatch.setattr(
        reverie, "persist_expectation_verdict", lambda *a: persisted.append(a)
    )
    bus = AsyncMock()
    await reverie._maybe_score_pending_expectation(bus)
    assert persisted == []


@pytest.mark.asyncio
async def test_scoring_malformed_pending_row_is_logged_not_silent(monkeypatch, caplog):
    """Defensive gap the review found: a row with a blank thought_id/expectation
    isn't reachable via the live write path today, but must not disappear
    silently -- with limit=1 it would otherwise keep resurfacing (still
    most-overdue) and starve every other real pending expectation forever with
    no trace of why."""
    from app import reverie
    from app.settings import settings

    monkeypatch.setattr(settings, "reverie_expectation_scoring_enabled", True)
    monkeypatch.setattr(
        reverie, "load_pending_expectations", lambda limit: [{"thought_id": "", "expectation": ""}]
    )
    persisted = []
    monkeypatch.setattr(reverie, "persist_expectation_verdict", lambda *a: persisted.append(a))
    bus = AsyncMock()
    with caplog.at_level("WARNING", logger="orion-thought.reverie"):
        await reverie._maybe_score_pending_expectation(bus)
    assert persisted == []  # nothing valid to persist against
    assert any("missing thought_id/expectation" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_scoring_pending_no_fresh_percept_persists_unscored_no_llm_call(monkeypatch):
    from app import reverie, vision_reader
    from app.settings import settings

    monkeypatch.setattr(settings, "reverie_expectation_scoring_enabled", True)
    monkeypatch.setattr(
        reverie,
        "load_pending_expectations",
        lambda limit: [{"thought_id": "t1", "expectation": "the door will still be closed"}],
    )
    monkeypatch.setattr(vision_reader, "read_recent_vision_events", lambda **kwargs: [])
    persisted = []
    monkeypatch.setattr(
        reverie, "persist_expectation_verdict", lambda tid, v, ts: persisted.append((tid, v))
    )
    cortex = AsyncMock()

    def boom(**kwargs):
        raise AssertionError("judge LLM call must not happen with no fresh percept")

    cortex.execute_plan = boom
    bus = AsyncMock()
    await reverie._maybe_score_pending_expectation(bus, cortex_client=cortex)
    assert persisted == [("t1", "unscored")]


@pytest.mark.asyncio
async def test_scoring_pending_fresh_percept_calls_judge_and_persists_verdict(monkeypatch):
    from app import reverie, vision_reader
    from app.settings import settings

    monkeypatch.setattr(settings, "reverie_expectation_scoring_enabled", True)
    monkeypatch.setattr(
        reverie,
        "load_pending_expectations",
        lambda limit: [{"thought_id": "t1", "expectation": "the door will still be closed"}],
    )
    monkeypatch.setattr(
        vision_reader,
        "read_recent_vision_events",
        lambda **kwargs: [{"narrative": "the door is closed", "age_sec": 5.0}],
    )
    persisted = []
    monkeypatch.setattr(
        reverie, "persist_expectation_verdict", lambda tid, v, ts: persisted.append((tid, v))
    )
    cortex = AsyncMock()
    cortex.execute_plan = AsyncMock(
        return_value={"final_text": json.dumps({"verdict": "confirmed"})}
    )
    bus = AsyncMock()
    await reverie._maybe_score_pending_expectation(bus, cortex_client=cortex)
    assert persisted == [("t1", "confirmed")]
    cortex.execute_plan.assert_awaited_once()
    plan_request = cortex.execute_plan.call_args.kwargs["req"]
    assert plan_request.context["expectation"] == "the door will still be closed"
    assert plan_request.context["percept_narrative"] == "the door is closed"


@pytest.mark.asyncio
async def test_scoring_judge_failure_persists_unscored_never_raises(monkeypatch):
    from app import reverie, vision_reader
    from app.settings import settings

    monkeypatch.setattr(settings, "reverie_expectation_scoring_enabled", True)
    monkeypatch.setattr(
        reverie,
        "load_pending_expectations",
        lambda limit: [{"thought_id": "t1", "expectation": "the door will still be closed"}],
    )
    monkeypatch.setattr(
        vision_reader,
        "read_recent_vision_events",
        lambda **kwargs: [{"narrative": "the door is closed", "age_sec": 5.0}],
    )
    persisted = []
    monkeypatch.setattr(
        reverie, "persist_expectation_verdict", lambda tid, v, ts: persisted.append((tid, v))
    )
    cortex = AsyncMock()
    cortex.execute_plan = AsyncMock(side_effect=RuntimeError("gateway timeout"))
    bus = AsyncMock()
    await reverie._maybe_score_pending_expectation(bus, cortex_client=cortex)  # must not raise
    assert persisted == [("t1", "unscored")]


@pytest.mark.asyncio
async def test_scoring_pending_load_failure_never_raises(monkeypatch):
    from app import reverie
    from app.settings import settings

    monkeypatch.setattr(settings, "reverie_expectation_scoring_enabled", True)

    def boom(limit):
        raise RuntimeError("db down")

    monkeypatch.setattr(reverie, "load_pending_expectations", boom)
    bus = AsyncMock()
    await reverie._maybe_score_pending_expectation(bus)  # must not raise


@pytest.mark.asyncio
async def test_run_reverie_once_calls_scoring_pass_before_narration(monkeypatch):
    """The scoring pass runs on every enabled tick, wired into the real
    run_reverie_once entrypoint (not just the helper in isolation) -- and
    genuinely runs *before* the narration LLM call, not merely "at some point"
    during the tick. A shared order list, appended to by both the scoring stub
    and the narration mock's own side_effect, is what actually proves the
    ordering claim in this test's name."""
    from app import reverie
    from app.settings import settings

    monkeypatch.setattr(settings, "reverie_expectation_scoring_enabled", True)
    order: list[str] = []

    async def _fake_score(bus, *, cortex_client=None):
        order.append("scoring")

    monkeypatch.setattr(reverie, "_maybe_score_pending_expectation", _fake_score)
    bus = AsyncMock()
    cortex = AsyncMock()

    async def _narrate_side_effect(**_kwargs):
        order.append("narration")
        return {"final_text": json.dumps({"interpretation": GROUNDED_TEXT, "evidence_refs": ["ol-1"]})}

    cortex.execute_plan = AsyncMock(side_effect=_narrate_side_effect)
    result = await reverie.run_reverie_once(
        bus, broadcast_reader=lambda: _broadcast(), cortex_client=cortex
    )
    assert result is not None
    assert order == ["scoring", "narration"]


@pytest.mark.asyncio
async def test_run_reverie_once_scoring_failure_does_not_block_narration(monkeypatch):
    from app import reverie
    from app.settings import settings

    monkeypatch.setattr(settings, "reverie_expectation_scoring_enabled", True)

    async def _boom(bus, *, cortex_client=None):
        raise RuntimeError("scoring blew up")

    monkeypatch.setattr(reverie, "_maybe_score_pending_expectation", _boom)
    bus = AsyncMock()
    cortex = AsyncMock()
    cortex.execute_plan = AsyncMock(
        return_value={"final_text": json.dumps({"interpretation": GROUNDED_TEXT, "evidence_refs": ["ol-1"]})}
    )
    result = await reverie.run_reverie_once(
        bus, broadcast_reader=lambda: _broadcast(), cortex_client=cortex
    )
    assert result is not None  # narration still succeeded despite scoring blowing up


# --- 8. run_reverie_once: expectation_checkable_by stamping --------------------


@pytest.mark.asyncio
async def test_new_expectation_gets_checkable_window_when_flag_on(monkeypatch):
    from app import reverie
    from app.settings import settings

    monkeypatch.setattr(settings, "reverie_expectation_scoring_enabled", True)
    monkeypatch.setattr(settings, "reverie_expectation_check_window_sec", 1800.0)
    monkeypatch.setattr(reverie, "_maybe_score_pending_expectation", AsyncMock())
    bus = AsyncMock()
    cortex = AsyncMock()
    cortex.execute_plan = AsyncMock(
        return_value={
            "final_text": json.dumps(
                {
                    "interpretation": GROUNDED_TEXT,
                    "evidence_refs": ["ol-1"],
                    "expectation": "the door will still be closed",
                }
            )
        }
    )
    before = datetime.now(timezone.utc)
    result = await reverie.run_reverie_once(
        bus, broadcast_reader=lambda: _broadcast(), cortex_client=cortex
    )
    after = datetime.now(timezone.utc)
    assert result is not None
    assert result.expectation == "the door will still be closed"
    assert result.expectation_checkable_by is not None
    lo = before + timedelta(seconds=1800.0)
    hi = after + timedelta(seconds=1800.0)
    assert lo <= result.expectation_checkable_by <= hi


@pytest.mark.asyncio
async def test_new_expectation_checkable_window_not_set_when_flag_off(monkeypatch):
    from app import reverie
    from app.settings import settings

    monkeypatch.setattr(settings, "reverie_expectation_scoring_enabled", False)
    bus = AsyncMock()
    cortex = AsyncMock()
    cortex.execute_plan = AsyncMock(
        return_value={
            "final_text": json.dumps(
                {
                    "interpretation": GROUNDED_TEXT,
                    "evidence_refs": ["ol-1"],
                    "expectation": "the door will still be closed",
                }
            )
        }
    )
    result = await reverie.run_reverie_once(
        bus, broadcast_reader=lambda: _broadcast(), cortex_client=cortex
    )
    assert result is not None
    # Narrate still captured the text (flag only gates the checkable window
    # and the periodic scoring scan, not the field itself) -- but with scoring
    # off, no window is ever opened on it.
    assert result.expectation == "the door will still be closed"
    assert result.expectation_checkable_by is None
