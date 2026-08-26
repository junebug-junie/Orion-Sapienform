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
# `foveal` must be UNAMBIGUOUSLY rank 1. Review finding 2026-08-26: the old
# line gave all five tokens identical counts, and the sort's final tiebreak is
# alphabetical -- so every tick test was silently investigating "again", and
# `test_the_prompt_names_the_term...` passed only because "foveal" also appears
# in the prompt's "also above their usual rate" line.
_RECENT_LINE = "foveal foveal foveal probe window sizing"   # foveal wins 3:1
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
        return (text or ""), {
            "elapsed_sec": 1.0,
            "harness_step_count": 12,
            "harness_grounding_status": "grounded",
        }

    loop._generate = _fake_generate  # type: ignore[assignment]
    return loop


def _live_messages():
    return _corpus(datetime.now(timezone.utc))


def test_the_correlation_id_is_uuid_shaped() -> None:
    """REGRESSION, first live deploy 2026-08-26. `BaseEnvelope.correlation_id`
    validates as a UUID, and `execute_unified_turn` builds envelopes
    internally -- so a readable `tag:term:ts` correlation id killed the whole
    turn on `uuid_parsing` before it started. The other tick tests stub
    `_generate` wholesale, so none of them touch envelope construction; this
    one pins the shape directly."""
    from uuid import UUID

    bus = _FakeBus()
    seen = {}

    loop = _loop(bus, text="found", messages=_live_messages())

    async def _capture(prompt, correlation_id):
        seen["corr"] = correlation_id
        return "found", {}

    loop._generate = _capture  # type: ignore[assignment]
    assert asyncio.run(loop.tick()) is None
    UUID(seen["corr"])  # raises if not a UUID
    assert UUID(bus.published[0][1].payload["correlation_id"])


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


def test_the_same_term_is_never_investigated_twice() -> None:
    """The invariant is "no term twice", NOT "the second tick blocks" -- since
    the rank-1 fallthrough landed, a marked top term means the loop moves to
    the next candidate rather than losing the day."""
    bus = _FakeBus()
    msgs = _live_messages()
    targets = []
    for _ in range(3):
        loop = _loop(bus, text="found something", messages=msgs, min_cooldown_sec=0.0)
        if asyncio.run(loop.tick()) is not None:
            break
        targets.append(bus.published[-1][1].payload["source_ref"])
    assert len(targets) == len(set(targets)), f"a term repeated: {targets}"
    assert targets[0] == "curiosity:foveal"


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
    loop._message_source = lambda: _corpus_for("kestrel")
    assert asyncio.run(loop.tick()) == "daily_cap"
    assert len(bus.published) == 1


# --- the target must be the term the tests think it is --------------------


def test_the_tick_targets_the_genuinely_top_term() -> None:
    """REGRESSION 2026-08-26: the fixture used to tie all five tokens at equal
    counts, and the sort's final tiebreak is alphabetical -- so every tick test
    was really exercising "again"."""
    bus = _FakeBus()
    loop = _loop(bus, text="found", messages=_live_messages())
    assert asyncio.run(loop.tick()) is None
    assert bus.published[0][1].payload["source_ref"] == "curiosity:foveal"
    assert bus.published[0][1].payload["title"] == "Curiosity: foveal"


# --- the lookup gate -------------------------------------------------------


def _loop_with_frame(bus, frame: dict, messages=None):
    loop = _loop(bus, text=None, messages=messages or _live_messages())

    async def _fake_turn(prompt, correlation_id):
        # Exercise the REAL _generate frame handling by calling it with a
        # stubbed execute_unified_turn result.
        return await CuriosityInvestigation._generate.__wrapped__(loop, prompt, correlation_id) \
            if hasattr(CuriosityInvestigation._generate, "__wrapped__") else ("", {})

    return loop


def test_a_turn_that_looked_nothing_up_is_not_journaled() -> None:
    """THE load-bearing gate. A turn that called no tools and simply wrote
    fluent prose from parametric knowledge produces a perfectly well-formed
    llm_response, and without this check lands in the journal
    indistinguishable from a real investigation -- CLAUDE.md 0A's
    no-empty-shell-cognition clause verbatim."""
    bus = _FakeBus()
    loop = _loop(bus, text="Foveal vision concerns the fovea centralis...", messages=_live_messages())

    async def _no_lookup(prompt, correlation_id):
        return "Foveal vision concerns the fovea centralis...", {
            "harness_step_count": 0,
            "harness_grounding_status": None,
        }

    # Simulate what the real _generate does with such a frame.
    async def _gen(prompt, correlation_id):
        text, debug = await _no_lookup(prompt, correlation_id)
        if debug["harness_step_count"] < MIN_HARNESS_STEPS:
            return "", {"error": "no_lookup", **debug}
        return text, debug

    loop._generate = _gen  # type: ignore[assignment]
    assert asyncio.run(loop.tick()) == "empty_generation"
    assert bus.published == [], "an ungrounded turn must never reach the journal"


def test_the_journal_records_the_lookup_evidence() -> None:
    """The claim "I investigated this" has to stay checkable in the artifact."""
    bus = _FakeBus()
    loop = _loop(bus, text="Here is what I found.", messages=_live_messages())
    assert asyncio.run(loop.tick()) is None
    body = bus.published[0][1].payload["body"]
    assert "12 harness steps" in body
    assert "grounded" in body


def test_the_lookup_bar_is_above_a_bare_answer() -> None:
    assert MIN_HARNESS_STEPS >= 3


# --- gates survive a restart ----------------------------------------------


def test_the_cooldown_and_daily_cap_survive_a_restart() -> None:
    """REGRESSION 2026-08-26: both were in-process fields, so six consecutive
    Hub restarts produced six journal entries against a cap of 3/day with 4h
    between. `.env_example` states that cap as a guarantee."""
    bus = _FakeBus()
    first = _loop(bus, text="found", messages=_live_messages())
    assert asyncio.run(first.tick()) is None
    assert len(bus.published) == 1

    # A brand-new object sharing the same Redis == a Hub restart.
    for _ in range(4):
        restarted = _loop(bus, text="found", messages=_live_messages())
        assert asyncio.run(restarted.tick()) == "cooldown"
    assert len(bus.published) == 1, "a redeploy is not a licence to investigate again"


def test_the_daily_cap_survives_a_restart_once_the_cooldown_lapses() -> None:
    bus = _FakeBus()
    for i in range(3):
        loop = _loop(bus, text="found", messages=_live_messages(), min_cooldown_sec=0.0)
        loop._message_source = lambda i=i: _corpus_for(f"kestrel{'x' * i}")
        assert asyncio.run(loop.tick()) is None
    over = _loop(bus, text="found", messages=_live_messages(), min_cooldown_sec=0.0)
    over._message_source = lambda: _corpus_for("gannet")
    assert asyncio.run(over.tick()) == "daily_cap"
    assert len(bus.published) == 3


def _corpus_for(word: str):
    now = datetime.now(timezone.utc)
    messages = [(now - timedelta(hours=2), f"{word} {word} {word} probe window sizing")] * 120
    messages += [(now - timedelta(days=5), _BASELINE_LINE)] * 400
    return messages


# --- a broken mount must be loud ------------------------------------------


def test_an_empty_corpus_is_reported_as_a_broken_mount_not_a_quiet_day() -> None:
    """`iter_all_human_messages` on a missing root returns [] without raising,
    so this reason is the ONLY place that failure is distinguishable."""
    bus = _FakeBus()
    loop = _loop(bus, text="found", messages=[])
    assert asyncio.run(loop.tick()) == "corpus_empty"
    assert bus.published == []


def test_a_thin_but_real_corpus_is_not_a_broken_mount() -> None:
    bus = _FakeBus()
    thin = [(datetime.now(timezone.utc) - timedelta(hours=2), "a few words only")] * 3
    loop = _loop(bus, text="found", messages=thin)
    assert asyncio.run(loop.tick()) == "corpus_underpowered"


# --- rank-1 marked must not lose the day ----------------------------------


def test_a_marked_top_term_falls_through_to_the_next_candidate() -> None:
    """REGRESSION 2026-08-26: replaying the real corpus day-by-day, 2 of 16
    otherwise-eligible days produced nothing because rank 1 was inside its own
    7-day mark -- including the single hottest day in the window."""
    bus = _FakeBus()
    first = _loop(bus, text="found", messages=_live_messages())
    assert asyncio.run(first.tick()) is None
    assert bus.published[0][1].payload["source_ref"] == "curiosity:foveal"

    second = _loop(bus, text="found", messages=_live_messages(), min_cooldown_sec=0.0)
    assert asyncio.run(second.tick()) is None
    assert bus.published[1][1].payload["source_ref"] != "curiosity:foveal", (
        "rank 1 was marked; the loop must fall through, not lose the day"
    )


def test_all_candidates_marked_still_blocks() -> None:
    bus = _FakeBus()
    msgs = _live_messages()
    for _ in range(8):
        loop = _loop(bus, text="found", messages=msgs, min_cooldown_sec=0.0, daily_cap=-1)
        if loop.tick.__name__ and asyncio.run(loop.tick()) == "term_already_investigated":
            break
    else:
        raise AssertionError("exhausting every candidate must eventually block")


# --- the real _generate, with execute_unified_turn stubbed ----------------
#
# Every other tick test replaces `_generate` wholesale, so nothing exercised
# what it sends or how it reads the frame back. These drive the real method.


def _drive_real_generate(frames, *, monkeypatch=None):
    """Call the real `_generate` against a stubbed `execute_unified_turn`,
    capturing the arguments it was invoked with."""
    import orion.hub.turn_orchestrator as orchestrator

    captured: dict = {}
    original = orchestrator.execute_unified_turn

    async def _stub(**kwargs):
        captured.update(kwargs)
        return frames

    orchestrator.execute_unified_turn = _stub
    try:
        bus = _FakeBus()
        loop = _loop(bus, text=None, messages=_live_messages())
        loop._generate = CuriosityInvestigation._generate.__get__(loop)
        result = asyncio.run(loop._generate("prompt text", "corr-1"))
    finally:
        orchestrator.execute_unified_turn = original
    return result, captured


def test_the_turn_is_sent_with_no_write_so_it_does_not_double_persist() -> None:
    """`_journal` is the sole persistence path. Without `no_write`, the
    governor also writes an untagged duplicate chat row."""
    frames = [{"type": "final", "llm_response": "found it", "harness_step_count": 9}]
    (text, debug), captured = _drive_real_generate(frames)
    assert text == "found it"
    assert captured["payload"]["no_write"] is True
    assert captured["payload"]["source"] == "curiosity_investigation"
    assert captured["session_id"] == "orion_curiosity"


def test_generate_refuses_a_turn_with_too_few_harness_steps() -> None:
    frames = [{"type": "final", "llm_response": "fluent but ungrounded", "harness_step_count": 1}]
    (text, debug), _ = _drive_real_generate(frames)
    assert text == ""
    assert debug["error"] == "no_lookup"


def test_generate_refuses_a_context_overflow() -> None:
    frames = [
        {
            "type": "final",
            "llm_response": "truncated",
            "harness_step_count": 20,
            "context_overflow": True,
        }
    ]
    (text, debug), _ = _drive_real_generate(frames)
    assert text == ""
    assert debug["error"] == "context_overflow"


def test_generate_refuses_error_shaped_text() -> None:
    frames = [
        {"type": "final", "llm_response": "Error: upstream failed", "harness_step_count": 20}
    ]
    (text, debug), _ = _drive_real_generate(frames)
    assert text == ""


def test_generate_treats_a_deferred_turn_as_silence_not_failure() -> None:
    """Thought declining an unsolicited turn is a legitimate outcome."""
    (text, debug), _ = _drive_real_generate([{"type": "turn_deferred", "reason": "busy"}])
    assert text == ""
    assert debug["error"] == "no_final_frame"
    assert debug["frame_type"] == "turn_deferred"


def test_generate_passes_a_grounded_turn_through_with_its_evidence() -> None:
    frames = [
        {
            "type": "final",
            "llm_response": "here is what I found",
            "harness_step_count": 29,
            "harness_grounding_status": "grounded",
        }
    ]
    (text, debug), _ = _drive_real_generate(frames)
    assert text == "here is what I found"
    assert debug["harness_step_count"] == 29
    assert debug["harness_grounding_status"] == "grounded"


# --- the in-process fallback when Redis is unavailable --------------------


def test_the_daily_counter_rolls_over_when_there_is_no_redis() -> None:
    """With Redis absent the loop falls back to in-process counting, so the
    UTC/local date rollover in `_roll_daily_counter` is the only thing keeping
    yesterday's count from capping today."""

    class _NoRedisBus(_FakeBus):
        def __init__(self) -> None:
            super().__init__()
            # AFTER super().__init__, which assigns a real fake redis --
            # setting this as a class attribute is silently shadowed by the
            # instance attribute and the test then exercises the Redis path it
            # meant to exclude.
            self.redis = None

    bus = _NoRedisBus()
    assert bus.redis is None, "fixture must genuinely have no redis"
    loop = _loop(bus, text="found", messages=_live_messages(), daily_cap=1)
    loop._done_today = 1
    loop._done_today_date = (datetime.now(timezone.utc) - timedelta(days=1)).date().isoformat()
    assert asyncio.run(loop.tick()) is None, "yesterday's count must not cap today"
    assert loop._done_today == 1
