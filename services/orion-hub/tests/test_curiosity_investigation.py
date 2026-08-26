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
from orion.curiosity.term_surfacing import build_surfacing_report
from scripts.curiosity_investigation import (
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
    base = dict(has_signal=True, underpowered=False, term_recently_investigated=False)
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
        ({"underpowered": True}, "corpus_underpowered"),
        ({"has_signal": False}, "no_surfaced_term"),
        ({"term_recently_investigated": True}, "term_already_investigated"),
    ],
)
def test_each_signal_gate_blocks_with_its_own_reason(over, expected) -> None:
    assert signal_block_reason(_signal(**over)) == expected


def test_the_corpus_is_not_read_when_a_cheap_gate_already_blocks() -> None:
    """Parsing the transcripts is ~7.8s of blocking IO over ~1.1 GB. Reading it
    on every tick would burn that to answer a question the cooldown already
    settles -- and would stall Hub's event loop while Hub serves real turns."""
    bus = _FakeBus()
    reads = {"n": 0}

    def _counting_source():
        reads["n"] += 1
        return _live_messages()

    loop = _loop(bus, text="found", messages=None, message_source=_counting_source)
    assert asyncio.run(loop.tick()) is None
    assert reads["n"] == 1
    # Second tick is inside the cooldown: it must not touch the corpus at all.
    assert asyncio.run(loop.tick()) == "cooldown"
    assert reads["n"] == 1


def test_a_thin_corpus_is_not_reported_as_a_quiet_day() -> None:
    """"Too little was said to tell" and "nothing stood out" are different
    claims. Collapsing them makes a broken transcript reader indistinguishable
    from an ordinary quiet day."""
    assert signal_block_reason(_signal(underpowered=True, has_signal=False)) == (
        "corpus_underpowered"
    )


def test_a_negative_daily_cap_disables_the_cap() -> None:
    assert scheduling_block_reason(_sched(daily_cap=-1, done_today=999)) is None


def test_the_first_ever_tick_is_not_blocked_by_cooldown() -> None:
    assert scheduling_block_reason(_sched(seconds_since_last=None)) is None


# --- journal entry ---------------------------------------------------------


# Both windows have to clear `SurfacingReport.underpowered` (200 recent / 2000
# baseline tokens) or every tick below blocks on `corpus_underpowered` before it
# reaches the gate under test. Caught by that guard on the first run, which is
# the guard doing its job.
_RECENT_LINE = "foveal probe window sizing again"          # 5 scoring tokens
_BASELINE_LINE = "assorted background chatter concerning unrelated topics"  # 6


def _corpus(now: datetime):
    messages = [(now - timedelta(hours=2), _RECENT_LINE)] * 120
    messages += [(now - timedelta(days=5), _BASELINE_LINE)] * 400
    return messages


def _report_and_target():
    report = build_surfacing_report(_corpus(NOW), now=NOW)
    assert not report.underpowered, "fixture must be measurable"
    assert report.terms, "fixture must surface something"
    return report, report.terms[0]


def test_the_journal_entry_keeps_the_finding_that_prompted_it() -> None:
    """An entry carrying only the conclusion is not inspectable -- a reader
    cannot tell what Orion was reacting to, or check the claim."""
    report, target = _report_and_target()
    entry = build_investigation_journal_entry(
        report=report, target=target, body_text="I looked and here is what I found.",
        correlation_id="corr-1", created_at=NOW,
    )
    assert "What I noticed:" in entry.body
    assert target.term in entry.body
    assert "separate messages" in entry.body
    assert "I looked and here is what I found." in entry.body


def test_the_journal_entry_is_namespaced_away_from_the_analysis_sources() -> None:
    """The self-study analysis cooldown matches `<source>:` prefixes on
    source_ref. A curiosity entry must not be able to suppress one of those."""
    report, target = _report_and_target()
    entry = build_investigation_journal_entry(
        report=report, target=target, body_text="x", correlation_id="c", created_at=NOW
    )
    assert entry.source_ref == f"curiosity:{target.term}"
    assert entry.source_kind == "self_study"
    for analysis_source in (
        "concept_induction", "vision_events", "affective_state", "cocreation_signals",
    ):
        assert not entry.source_ref.startswith(f"{analysis_source}:")


# --- one real tick, with fakes ---------------------------------------------


class _FakeRedis:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}

    async def get(self, key):
        return self.values.get(key)

    async def setex(self, key, ttl, value):
        self.values[key] = value


class _FakeBus:
    def __init__(self) -> None:
        self.redis = _FakeRedis()
        self.published: list = []

    async def publish(self, channel, envelope):
        self.published.append((channel, envelope))


def _loop(bus, *, text: str | None, messages, **over) -> CuriosityInvestigation:
    kwargs = dict(
        enabled=True, tick_interval_sec=60.0, min_cooldown_sec=14400.0, daily_cap=3,
        timeout_sec=300.0, session_id="orion_curiosity", term_mark_ttl_sec=604800,
        recent_hours=24.0, baseline_days=14.0,
        message_source=(lambda: messages), source_ref=SOURCE,
    )
    kwargs.update(over)
    loop = CuriosityInvestigation(**kwargs)
    loop._bus = bus
    loop._harness_rpc_bus = bus

    async def _fake_generate(prompt, correlation_id):
        loop.seen_prompt = prompt
        return (text or ""), {"elapsed_sec": 1.0}

    loop._generate = _fake_generate  # type: ignore[assignment]
    return loop


def _live_messages():
    return _corpus(datetime.now(timezone.utc))


def test_a_successful_tick_journals_exactly_one_entry() -> None:
    bus = _FakeBus()
    loop = _loop(bus, text="Here is what I found when I looked.", messages=_live_messages())
    assert asyncio.run(loop.tick()) is None
    assert len(bus.published) == 1
    channel, envelope = bus.published[0]
    assert channel == "orion:journal:write"
    assert envelope.kind == "journal.entry.write.v1"
    assert envelope.payload["source_ref"].startswith("curiosity:")


def test_a_deferred_or_empty_turn_writes_nothing() -> None:
    """Silence over a false positive: if the turn produced no text, there is
    nothing Orion actually found, and the journal must stay clean."""
    bus = _FakeBus()
    loop = _loop(bus, text="", messages=_live_messages())
    assert asyncio.run(loop.tick()) == "empty_generation"
    assert bus.published == []


def test_a_failed_turn_still_consumes_its_slot() -> None:
    """Otherwise a term that reliably fails is retried every tick forever."""
    bus = _FakeBus()
    loop = _loop(bus, text="", messages=_live_messages())
    asyncio.run(loop.tick())
    assert loop._done_today == 1
    assert bus.redis.values, "the term must be marked even though the turn failed"


def test_the_same_term_is_not_investigated_twice() -> None:
    bus = _FakeBus()
    msgs = _live_messages()
    first = _loop(bus, text="found something", messages=msgs)
    assert asyncio.run(first.tick()) is None
    second = _loop(bus, text="found something", messages=msgs)
    assert asyncio.run(second.tick()) == "term_already_investigated"
    assert len(bus.published) == 1


def test_an_ordinary_day_investigates_nothing() -> None:
    bus = _FakeBus()
    line = "assorted background chatter concerning unrelated topics"
    same = [(datetime.now(timezone.utc) - timedelta(hours=2), line)] * 120
    same += [(datetime.now(timezone.utc) - timedelta(days=5), line)] * 400
    loop = _loop(bus, text="should never be used", messages=same)
    assert asyncio.run(loop.tick()) in ("no_surfaced_term", "corpus_underpowered")
    assert bus.published == []


def test_the_prompt_names_the_term_and_licenses_a_null_result() -> None:
    """Both properties are load-bearing: without the term there is nothing to
    look up, and without permission to find nothing the loop manufactures
    significance every single day."""
    bus = _FakeBus()
    loop = _loop(bus, text="ok", messages=_live_messages())
    asyncio.run(loop.tick())
    prompt = loop.seen_prompt
    assert "foveal" in prompt
    assert "nothing worth noting" in prompt
    assert "search your own recall" in prompt
    assert "Nobody asked you for this" in prompt


def test_the_daily_cap_is_enforced_across_ticks() -> None:
    bus = _FakeBus()
    loop = _loop(bus, text="found", messages=_live_messages(), daily_cap=1, min_cooldown_sec=0.0)
    assert asyncio.run(loop.tick()) is None
    loop._message_source = lambda: (
        [(datetime.now(timezone.utc) - timedelta(hours=2), "kestrel sighting circling overhead again")] * 120
        + [(datetime.now(timezone.utc) - timedelta(days=5), _BASELINE_LINE)] * 400
    )
    assert asyncio.run(loop.tick()) == "daily_cap"
    assert len(bus.published) == 1
