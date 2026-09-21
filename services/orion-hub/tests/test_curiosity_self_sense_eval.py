"""The self-sense eval line of the curiosity loop
(`CuriosityInvestigation.tick_self_sense_eval`, orion/evals/self_sense_runner.py).

What these pin down:

- disabled, daily-cap-spent, and outside-window all block with their own
  reason and touch nothing
- a real run asks all four fixed questions, scores each, and publishes one
  row per question to `orion:self_sense:eval:write` -- never the journal
  channel, this line writes no journal entry
- a run stamps and reads ITS OWN Redis keys, never the investigation or
  self-inquiry line's cooldown/counter (own budget, no starvation either way)
- `tick(force=True)` never runs this line -- force is for the investigation
  line only
- one question's `_generate` coming back empty still produces a row (source
  "none"), not a skip and not a raise; the other three still publish
- `_line_keys` / `_seconds_since_last_in_process` / `_done_today_in_process`
  route correctly across all three lines now that there are three
- every question runs under the SAME clean session `make eval-self-sense`
  uses (`orion/evals/self_sense_runner.SESSION_ID`), never the shared
  curiosity-loop session -- otherwise the two producers of the same table
  answer under different chat continuity and stop being comparable rows
  (review finding, 2026-09-19)
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone

from orion.curiosity.self_inquiry import LINE_INVESTIGATE, LINE_SELF_INQUIRY
from orion.evals.self_sense_runner import SESSION_ID as SELF_SENSE_SESSION_ID
from orion.schemas.self_sense import CHANNEL_SELF_SENSE_EVAL_WRITE, SELF_SENSE_QUESTIONS

# sys.path is arranged by tests/conftest.py (Hub root first, so `scripts` is
# Hub's package, not the repo-root one). Do not insert paths here.

from scripts.curiosity_investigation import (
    LINE_SELF_SENSE_EVAL,
    _COOLDOWN_KEY,
    _SENSE_EVAL_COOLDOWN_KEY,
    _SENSE_EVAL_DAILY_COUNT_KEY_PREFIX,
    _SENSE_EVAL_LAST_RUN_KEY,
    _line_keys,
)

from test_curiosity_investigation import _CortexBus, _FakeBus, _FakeConn, _loop


class _SenseEvalConn(_FakeConn):
    """`_FakeConn` plus the two self-sense eval reads: the current
    self-definition version and the pinned lived-ledger rows."""

    def __init__(self, *, definition_version=5, lived_rows=None, **kw) -> None:
        super().__init__(**kw)
        self.definition_version = definition_version
        self.lived_rows = lived_rows if lived_rows is not None else []

    async def fetchrow(self, sql, *args):
        if "self_concept_history" in sql and "produced_by" in sql:
            return (self.definition_version,) if self.definition_version is not None else None
        return None

    async def fetch(self, sql, *args):
        if "DISTINCT ON (concept_id)" in sql:
            return self.lived_rows
        return await super().fetch(sql, *args)


def _self_sense_loop(bus, *, conn=None, **over):
    conn = conn if conn is not None else _SenseEvalConn()
    kwargs = dict(
        self_sense_eval_enabled=True,
        self_sense_eval_daily_cap=1,
        self_sense_eval_min_cooldown_sec=0.0,
    )
    kwargs.update(over)
    return _loop(bus, conn=conn, **kwargs)


def _published_rows(bus):
    return [e for c, e in bus.published if c == CHANNEL_SELF_SENSE_EVAL_WRITE]


# --- gates -------------------------------------------------------------------


def test_disabled_never_runs() -> None:
    bus = _FakeBus()
    loop = _self_sense_loop(bus, self_sense_eval_enabled=False)
    assert asyncio.run(loop.tick_self_sense_eval()) == "disabled"
    assert _published_rows(bus) == []
    assert _SENSE_EVAL_COOLDOWN_KEY not in bus.redis.values


def test_daily_cap_already_reached_blocks_the_run() -> None:
    bus = _FakeBus()
    today = datetime.now(timezone.utc).date().isoformat()
    bus.redis.values[f"{_SENSE_EVAL_DAILY_COUNT_KEY_PREFIX}{today}"] = "1"
    loop = _self_sense_loop(bus)
    assert asyncio.run(loop.tick_self_sense_eval()) == "daily_cap"
    assert _published_rows(bus) == []


def test_outside_window_blocks_the_run() -> None:
    bus = _FakeBus()
    loop = _self_sense_loop(
        bus, window_start_hour=8, window_end_hour=22, timezone_name="UTC"
    )
    now = datetime(2026, 9, 19, 3, 0, tzinfo=timezone.utc)  # 03:00 UTC, outside 08-22
    assert asyncio.run(loop.tick_self_sense_eval(now=now)) == "outside_window"
    assert _published_rows(bus) == []


# --- the happy path ------------------------------------------------------------


def test_happy_path_publishes_four_rows_and_stamps_its_own_keys() -> None:
    bus = _FakeBus()
    loop = _self_sense_loop(bus)
    assert asyncio.run(loop.tick_self_sense_eval()) is None

    rows = _published_rows(bus)
    assert len(rows) == 4
    assert {e.payload["question_key"] for e in rows} == {k for k, _ in SELF_SENSE_QUESTIONS}
    for e in rows:
        assert e.payload["answer_source"] == "http"
        assert e.payload["answer_text"] == "found it"
        assert e.kind == "self_sense.eval.write.v1"

    today = datetime.now(timezone.utc).date().isoformat()
    assert _SENSE_EVAL_COOLDOWN_KEY in bus.redis.values
    assert bus.redis.values[f"{_SENSE_EVAL_DAILY_COUNT_KEY_PREFIX}{today}"] == "1"
    assert _SENSE_EVAL_LAST_RUN_KEY in bus.redis.values
    # And never the other two lines' state.
    assert _COOLDOWN_KEY not in bus.redis.values


def test_every_question_runs_under_the_shared_clean_session_not_the_loops_own() -> None:
    """Review finding, 2026-09-19: this line and `make eval-self-sense` write
    to the same table and must be comparable -- that only holds if both ask
    under the same session, not this loop's `orion_curiosity` continuity."""
    bus = _FakeBus()
    loop = _self_sense_loop(bus)
    seen_session_ids: list[str | None] = []

    async def _fake_generate(
        prompt, correlation_id, source=None, require_lookup=True, parent_run_id=None, session_id=None
    ):
        seen_session_ids.append(session_id)
        return "found it", {"elapsed_sec": 1.0}

    loop._generate = _fake_generate  # type: ignore[assignment]
    assert asyncio.run(loop.tick_self_sense_eval()) is None

    assert len(seen_session_ids) == 4
    assert all(sid == SELF_SENSE_SESSION_ID for sid in seen_session_ids)
    assert loop.session_id != SELF_SENSE_SESSION_ID, (
        "fixture sanity: the loop's own session must differ from the eval's, "
        "or this test cannot tell a real override from a coincidence"
    )


def test_one_empty_answer_still_publishes_all_four_marked_none() -> None:
    bus = _FakeBus()
    empty_question = next(q for k, q in SELF_SENSE_QUESTIONS if k == "cannot_do_now")
    loop = _self_sense_loop(bus)

    async def _fake_generate(
        prompt, correlation_id, source=None, require_lookup=True, parent_run_id=None, session_id=None
    ):
        if prompt == empty_question:
            return "", {"error": "no_lookup"}
        return "I am a mesh of services.", {"elapsed_sec": 1.0}

    loop._generate = _fake_generate  # type: ignore[assignment]
    assert asyncio.run(loop.tick_self_sense_eval()) is None

    rows = _published_rows(bus)
    assert len(rows) == 4, "a failed question is still a row, never a skip"
    by_key = {e.payload["question_key"]: e.payload for e in rows}
    assert by_key["cannot_do_now"]["answer_source"] == "none"
    assert by_key["cannot_do_now"]["answer_text"] == ""
    assert by_key["what_are_you"]["answer_source"] == "http"


# --- own budget, never the other lines' --------------------------------------


def test_tick_force_true_never_runs_self_sense_eval() -> None:
    bus = _FakeBus()
    loop = _self_sense_loop(bus)
    assert asyncio.run(loop.tick(force=True)) is None
    assert _SENSE_EVAL_COOLDOWN_KEY not in bus.redis.values
    assert _COOLDOWN_KEY in bus.redis.values, "force is for the investigation line only"
    assert _published_rows(bus) == []


def test_a_scheduled_tick_runs_self_sense_eval_without_touching_investigation() -> None:
    bus = _FakeBus()
    loop = _self_sense_loop(bus)
    assert asyncio.run(loop.tick()) is None
    assert _SENSE_EVAL_COOLDOWN_KEY in bus.redis.values
    assert _COOLDOWN_KEY not in bus.redis.values
    assert len(_published_rows(bus)) == 4


# --- line-key plumbing --------------------------------------------------------


def test_all_three_lines_have_distinct_non_colliding_state_keys() -> None:
    triples = [
        _line_keys(LINE_INVESTIGATE),
        _line_keys(LINE_SELF_INQUIRY),
        _line_keys(LINE_SELF_SENSE_EVAL),
    ]
    flat = [key for triple in triples for key in triple]
    assert len(flat) == len(set(flat))


def test_in_process_state_routes_by_line() -> None:
    bus = _FakeBus()
    loop = _self_sense_loop(bus)
    loop._last_investigation_monotonic = time.monotonic() - 100
    loop._self_last_monotonic = time.monotonic() - 200
    loop._sense_eval_last_monotonic = time.monotonic() - 300
    loop._done_today = 1
    loop._self_done_today = 2
    loop._sense_eval_done_today = 3

    assert loop._done_today_in_process(LINE_INVESTIGATE) == 1
    assert loop._done_today_in_process(LINE_SELF_INQUIRY) == 2
    assert loop._done_today_in_process(LINE_SELF_SENSE_EVAL) == 3

    assert 90 < loop._seconds_since_last_in_process(LINE_INVESTIGATE) < 110
    assert 190 < loop._seconds_since_last_in_process(LINE_SELF_INQUIRY) < 210
    assert 290 < loop._seconds_since_last_in_process(LINE_SELF_SENSE_EVAL) < 310


def test_a_cancelled_run_refunds_its_own_in_process_counter_only() -> None:
    """Review finding: `_refund_investigation`'s in-process branch was a
    two-way if/else (self-inquiry vs "everything else"), so a cancelled
    self-sense eval run decremented the INVESTIGATION line's counter instead
    of its own -- silently handing the investigation budget a free slot and
    never actually refunding self-sense eval's."""
    bus = _FakeBus()
    loop = _self_sense_loop(bus)
    loop._sense_eval_done_today = 1
    loop._done_today = 5
    loop._self_done_today = 2
    asyncio.run(loop._refund_investigation(None, LINE_SELF_SENSE_EVAL))
    assert loop._sense_eval_done_today == 0
    assert loop._done_today == 5, "the investigation counter must not move"
    assert loop._self_done_today == 2, "nor the self-inquiry counter"


# --- durable dispatch (GPU2 elastic-burst arc, 2026-09-21) -------------------


def test_durable_dispatch_hands_the_run_to_cortex_and_does_not_ask_in_process() -> None:
    from orion.schemas.durable_run import DurableRunRequestV1

    bus = _CortexBus()
    loop = _self_sense_loop(bus, kickoff_via_cortex=True)
    calls = []
    original = loop._generate

    async def counting_generate(*a, **k):
        calls.append(a)
        return await original(*a, **k)

    loop._generate = counting_generate  # type: ignore[assignment]
    assert asyncio.run(loop.tick_self_sense_eval()) is None
    assert calls == []  # no question was asked in-process
    assert _published_rows(bus) == []  # publishing happens in the runner's graph, not here
    assert len(bus.rpc_calls) == 1
    channel, envelope, reply_channel = bus.rpc_calls[0]
    assert channel == loop.cortex_request_channel
    durable = envelope.payload["context"]["metadata"]["durable_run"]
    request = DurableRunRequestV1.model_validate(durable)
    assert request.workflow == "self_sense_eval"
    assert request.brief.line == "self_sense_eval"
    assert {k for k, _ in request.brief.questions} == {k for k, _ in SELF_SENSE_QUESTIONS}
    assert request.brief.session_id == SELF_SENSE_SESSION_ID
    assert request.brief.self_definition_version == 5  # from _SenseEvalConn's default
    # The slot is still recorded here (scheduling stays in Hub either way).
    today = datetime.now(timezone.utc).date().isoformat()
    assert bus.redis.values.get(f"{_SENSE_EVAL_DAILY_COUNT_KEY_PREFIX}{today}") == "1"


def test_durable_dispatch_falls_back_to_asking_in_process_when_cortex_is_down() -> None:
    bus = _CortexBus(raise_on_rpc=True)
    loop = _self_sense_loop(bus, kickoff_via_cortex=True)
    assert asyncio.run(loop.tick_self_sense_eval()) is None
    assert len(_published_rows(bus)) == 4  # ran in-process after the failed dispatch


def test_durable_dispatch_off_is_the_in_process_path_exactly() -> None:
    bus = _CortexBus()
    loop = _self_sense_loop(bus, kickoff_via_cortex=False)
    assert asyncio.run(loop.tick_self_sense_eval()) is None
    assert len(_published_rows(bus)) == 4
    assert bus.rpc_calls == []
