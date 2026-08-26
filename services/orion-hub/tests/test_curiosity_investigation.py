"""Gates and one full tick for Orion's curiosity loop.

Same emphasis as the detector's tests: the REFUSALS are the product. This loop
drives a real unified turn and writes to Orion's journal, so every path that
must not do that is worth pinning.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from orion.core.bus.bus_schemas import ServiceRef
from orion.curiosity.study_material import StudyMaterial, assemble_study_material
from scripts.curiosity_investigation import (
    MIN_HARNESS_STEPS,
    CuriosityInvestigation,
    SchedulingGateInputs,
    SignalGateInputs,
    build_investigation_journal_entry,
    scheduling_block_reason,
    signal_block_reason,
)

NOW = datetime(2026, 8, 26, 12, 0, tzinfo=timezone.utc)
SOURCE = ServiceRef(name="orion-hub", version="0.1.0", node="athena")


def _sched(**over) -> SchedulingGateInputs:
    base = dict(
        enabled=True, seconds_since_last=None, min_cooldown_sec=14400.0,
        done_today=0, daily_cap=3,
    )
    base.update(over)
    return SchedulingGateInputs(**base)


def _signal(**over) -> SignalGateInputs:
    base = dict(has_material=True, stores_unavailable=False)
    base.update(over)
    return SignalGateInputs(**base)


# --- gates -----------------------------------------------------------------


def test_a_clear_tick_is_allowed() -> None:
    assert scheduling_block_reason(_sched()) is None
    assert signal_block_reason(_signal()) is None


@pytest.mark.parametrize(
    "over,expected",
    [
        ({"enabled": False}, "disabled"),
        ({"done_today": 3}, "daily_cap"),
        ({"seconds_since_last": 60.0}, "cooldown"),
    ],
)
def test_each_scheduling_gate_blocks_with_its_own_reason(over, expected) -> None:
    assert scheduling_block_reason(_sched(**over)) == expected


@pytest.mark.parametrize(
    "over,expected",
    [
        ({"stores_unavailable": True}, "stores_unavailable"),
        ({"has_material": False}, "no_approved_material"),
    ],
)
def test_each_signal_gate_blocks_with_its_own_reason(over, expected) -> None:
    assert signal_block_reason(_signal(**over)) == expected


def test_an_unreadable_store_is_not_reported_as_an_empty_mind() -> None:
    """An unreadable store and a mind with nothing in it must never be the same
    state: the only symptom of the former would be an absence of journal
    entries, which is also what a quiet stretch looks like."""
    assert signal_block_reason(_signal(stores_unavailable=True, has_material=False)) == (
        "stores_unavailable"
    )


def test_there_is_no_already_studied_gate() -> None:
    """Deliberate. Code no longer knows the subject -- Orion chooses it inside
    the turn -- so a "don't repeat" lock would be code guessing at Orion's own
    choice. It is shown what it recently studied and may repeat if it wants."""
    assert set(SignalGateInputs.__dataclass_fields__) == {
        "has_material", "stores_unavailable"
    }


def test_a_negative_daily_cap_disables_the_cap() -> None:
    assert scheduling_block_reason(_sched(daily_cap=-1, done_today=999)) is None


def test_the_first_ever_tick_is_not_blocked_by_cooldown() -> None:
    assert scheduling_block_reason(_sched(seconds_since_last=None)) is None


# --- fakes -----------------------------------------------------------------


class _FakeRedis:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}

    async def get(self, key):
        return self.values.get(key)

    async def setex(self, key, ttl, value):
        self.values[key] = value

    async def incr(self, key):
        self.values[key] = str(int(self.values.get(key, 0)) + 1)
        return int(self.values[key])

    async def expire(self, key, ttl):
        return True


class _FakeBus:
    def __init__(self) -> None:
        self.redis = _FakeRedis()
        self.published: list = []

    async def publish(self, channel, envelope):
        self.published.append((channel, envelope))


def _row(cid, kind="semantic", subject="a real thought"):
    return {
        "crystallization_id": cid, "kind": kind, "subject": subject,
        "summary": subject, "salience": 0.6, "created_at": NOW,
    }


def _relation_row(did):
    return {
        "decision_id": did, "relation": "same", "confidence": 0.95,
        "candidate_crystallization_id": "crys_dead", "target_crystallization_id": "t1",
        "decided_at": NOW, "candidate_subject": "", "candidate_summary": "",
        "target_subject": "the other one", "target_summary": "",
    }


class _FakeConn:
    def __init__(self, *, rows=None, relations=None, raises=False) -> None:
        self.rows = rows if rows is not None else [_row(f"c{i}") for i in range(4)]
        self.relations = relations if relations is not None else [_relation_row("d1")]
        self.raises = raises

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def fetch(self, sql, *args):
        if self.raises:
            raise RuntimeError("relation \"memory_crystallizations\" does not exist")
        if "GROUP BY kind" in sql:
            return [{"kind": "semantic", "n": 268}]
        if "FROM memory_crystallizations" in sql and "random()" in sql:
            return self.rows
        if "GROUP BY relation" in sql:
            return [{"relation": "same", "n": 316}]
        if "memory_concept_relation_decisions" in sql:
            return self.relations
        if "journal_entries" in sql:
            return []
        return []

    async def fetchval(self, sql, *args):
        if self.raises:
            raise RuntimeError("boom")
        return 356


class _FakePool:
    def __init__(self, conn) -> None:
        self._conn = conn

    def acquire(self):
        return self._conn


def _loop(bus, *, text: str | None = "found it", conn=None, **over) -> CuriosityInvestigation:
    conn = conn if conn is not None else _FakeConn()
    kwargs = dict(
        enabled=True, tick_interval_sec=60.0, min_cooldown_sec=14400.0, daily_cap=3,
        timeout_sec=1500.0, session_id="orion_curiosity",
        crystallization_sample=12, relation_sample=6,
        pool_provider=lambda: _FakePool(conn), source_ref=SOURCE,
    )
    kwargs.update(over)
    loop = CuriosityInvestigation(**kwargs)
    loop._bus = bus
    loop._harness_rpc_bus = bus

    async def _fake_generate(prompt, correlation_id):
        loop.seen_prompt = prompt
        return (text or ""), {
            "elapsed_sec": 1.0,
            "harness_step_count": 14,
            "harness_grounding_status": "grounded",
        }

    loop._generate = _fake_generate  # type: ignore[assignment]
    return loop


# --- one real tick ---------------------------------------------------------


def test_a_successful_tick_journals_exactly_one_entry() -> None:
    bus = _FakeBus()
    loop = _loop(bus)
    assert asyncio.run(loop.tick()) is None
    assert len(bus.published) == 1
    channel, envelope = bus.published[0]
    assert channel == "orion:journal:write"
    assert envelope.payload["source_kind"] == "self_study"
    assert envelope.payload["source_ref"].startswith("curiosity:")


def test_the_journal_title_does_not_invent_a_subject() -> None:
    """Code does not know what Orion chose -- it is in the prose. Deriving a
    title here would mean re-inferring that choice with a heuristic, which is
    the exact move this rewrite exists to delete."""
    bus = _FakeBus()
    assert asyncio.run(_loop(bus).tick()) is None
    assert bus.published[0][1].payload["title"] == "Curiosity"


def test_the_journal_records_what_was_offered() -> None:
    """So a reader can tell what was on the table when Orion chose."""
    bus = _FakeBus()
    assert asyncio.run(_loop(bus).tick()) is None
    body = bus.published[0][1].payload["body"]
    assert "Offered 4 of 268 approved concepts" in body
    assert "sampled at random" in body
    assert "14 harness steps" in body


def test_the_prompt_carries_real_material_and_no_assignment() -> None:
    bus = _FakeBus()
    loop = _loop(bus)
    asyncio.run(loop.tick())
    prompt = loop.seen_prompt
    assert "a real thought" in prompt
    assert "Nobody asked you" in prompt
    assert "quiet answer is a real answer" in prompt


def test_an_unreadable_store_writes_nothing() -> None:
    bus = _FakeBus()
    loop = _loop(bus, conn=_FakeConn(raises=True))
    assert asyncio.run(loop.tick()) == "stores_unavailable"
    assert bus.published == []


def test_a_missing_pool_writes_nothing() -> None:
    bus = _FakeBus()
    loop = _loop(bus, pool_provider=lambda: None)
    assert asyncio.run(loop.tick()) == "stores_unavailable"
    assert bus.published == []


def test_no_approved_material_writes_nothing() -> None:
    bus = _FakeBus()
    loop = _loop(bus, conn=_FakeConn(rows=[], relations=[]))
    assert asyncio.run(loop.tick()) == "no_approved_material"
    assert bus.published == []


def test_a_deferred_or_empty_turn_writes_nothing() -> None:
    bus = _FakeBus()
    loop = _loop(bus, text="")
    assert asyncio.run(loop.tick()) == "empty_generation"
    assert bus.published == []


def test_a_failed_turn_still_consumes_its_slot() -> None:
    """Otherwise a reliably failing turn is retried every tick forever."""
    bus = _FakeBus()
    loop = _loop(bus, text="")
    asyncio.run(loop.tick())
    assert loop._done_today == 1
    assert bus.redis.values.get("orion:curiosity:last_investigation_at")


def test_the_stores_are_not_read_when_a_cheap_gate_already_blocks() -> None:
    bus = _FakeBus()
    reads = {"n": 0}

    class _CountingConn(_FakeConn):
        async def fetch(self, sql, *args):
            reads["n"] += 1
            return await super().fetch(sql, *args)

    conn = _CountingConn()
    loop = _loop(bus, conn=conn)
    assert asyncio.run(loop.tick()) is None
    first = reads["n"]
    assert first > 0
    assert asyncio.run(loop.tick()) == "cooldown"
    assert reads["n"] == first, "a cooldown-blocked tick must not touch the stores"


# --- gates survive a restart ----------------------------------------------


def test_the_cooldown_survives_a_restart() -> None:
    """A redeploy is not a licence to run again."""
    bus = _FakeBus()
    assert asyncio.run(_loop(bus).tick()) is None
    for _ in range(4):
        assert asyncio.run(_loop(bus).tick()) == "cooldown"
    assert len(bus.published) == 1


def test_the_daily_cap_survives_a_restart() -> None:
    bus = _FakeBus()
    for _ in range(3):
        assert asyncio.run(_loop(bus, min_cooldown_sec=0.0).tick()) is None
    assert asyncio.run(_loop(bus, min_cooldown_sec=0.0).tick()) == "daily_cap"
    assert len(bus.published) == 3


def test_the_daily_counter_rolls_over_when_there_is_no_redis() -> None:
    class _NoRedisBus(_FakeBus):
        def __init__(self) -> None:
            super().__init__()
            self.redis = None

    bus = _NoRedisBus()
    assert bus.redis is None, "fixture must genuinely have no redis"
    loop = _loop(bus, daily_cap=1)
    loop._done_today = 1
    loop._done_today_date = (NOW - timedelta(days=1)).date().isoformat()
    assert asyncio.run(loop.tick()) is None, "yesterday's count must not cap today"


# --- the correlation id ----------------------------------------------------


def test_the_correlation_id_is_uuid_shaped() -> None:
    """`BaseEnvelope.correlation_id` validates as a UUID and
    `execute_unified_turn` builds envelopes internally, so a readable
    `tag:run:ts` string kills the turn on `uuid_parsing` before it starts."""
    from uuid import UUID

    bus = _FakeBus()
    seen = {}
    loop = _loop(bus)

    async def _capture(prompt, correlation_id):
        seen["corr"] = correlation_id
        return "found", {"harness_step_count": 9}

    loop._generate = _capture  # type: ignore[assignment]
    assert asyncio.run(loop.tick()) is None
    UUID(seen["corr"])
    assert UUID(bus.published[0][1].payload["correlation_id"])


# --- the real _generate, with execute_unified_turn stubbed ----------------


def _drive_real_generate(frames):
    import orion.hub.turn_orchestrator as orchestrator

    captured: dict = {}
    original = orchestrator.execute_unified_turn

    async def _stub(**kwargs):
        captured.update(kwargs)
        return frames

    orchestrator.execute_unified_turn = _stub
    try:
        loop = _loop(_FakeBus(), text=None)
        loop._generate = CuriosityInvestigation._generate.__get__(loop)
        result = asyncio.run(loop._generate("prompt text", "corr-1"))
    finally:
        orchestrator.execute_unified_turn = original
    return result, captured


def test_the_turn_is_sent_with_no_write_so_it_does_not_double_persist() -> None:
    frames = [{"type": "final", "llm_response": "found it", "harness_step_count": 9}]
    (text, debug), captured = _drive_real_generate(frames)
    assert text == "found it"
    assert captured["payload"]["no_write"] is True
    assert captured["payload"]["source"] == "curiosity_investigation"


def test_a_turn_that_looked_nothing_up_is_refused() -> None:
    """THE load-bearing gate. A turn that called no tools and wrote fluent
    prose from parametric knowledge produces a well-formed llm_response and,
    without this, lands in the journal indistinguishable from a real one."""
    frames = [{"type": "final", "llm_response": "fluent but ungrounded", "harness_step_count": 1}]
    (text, debug), _ = _drive_real_generate(frames)
    assert text == ""
    assert debug["error"] == "no_lookup"


def test_generate_refuses_a_context_overflow() -> None:
    frames = [{"type": "final", "llm_response": "x", "harness_step_count": 20, "context_overflow": True}]
    (text, debug), _ = _drive_real_generate(frames)
    assert text == "" and debug["error"] == "context_overflow"


def test_generate_refuses_error_shaped_text() -> None:
    frames = [{"type": "final", "llm_response": "Error: upstream failed", "harness_step_count": 20}]
    (text, _debug), _ = _drive_real_generate(frames)
    assert text == ""


def test_generate_treats_a_deferred_turn_as_silence_not_failure() -> None:
    """Thought declining an unsolicited turn is a legitimate outcome."""
    (text, debug), _ = _drive_real_generate([{"type": "turn_deferred", "reason": "busy"}])
    assert text == "" and debug["frame_type"] == "turn_deferred"


def test_the_lookup_bar_is_above_a_bare_answer() -> None:
    assert MIN_HARNESS_STEPS >= 3


# --- journal builder -------------------------------------------------------


def _material():
    return assemble_study_material(
        now=NOW,
        approved_counts=[{"kind": "semantic", "n": 268}],
        approved_rows=[_row("c1")],
        relation_counts=[{"relation": "same", "n": 316}],
        relation_rows=[_relation_row("d1")],
        relation_resolvable=356,
        recent_titles=[],
    )


def test_the_journal_entry_is_namespaced_away_from_the_analysis_sources() -> None:
    entry = build_investigation_journal_entry(
        material=_material(), body_text="x", correlation_id="c", run_id="r1", created_at=NOW
    )
    assert entry.source_ref == "curiosity:r1"
    for analysis_source in (
        "concept_induction", "vision_events", "affective_state", "cocreation_signals",
    ):
        assert not entry.source_ref.startswith(f"{analysis_source}:")
