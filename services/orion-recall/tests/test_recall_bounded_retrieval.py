"""Bounded retrieval (docs/superpowers/specs/2026-09-29-recall-retrieval-query-architecture-design.md).

Acceptance checks 1-4 of the design plus the Phase 2 intake fields. The long
query is the real 30,663-char stance_react prompt from recall_telemetry
(evals/fixtures/), the one that produced 268 sub-queries and a 72s recall.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import pytest

from app import worker
from app.storage import falkor_chat_adapter
from orion.core.contracts.recall import MemoryBundleV1, RecallQueryV1

FIXTURE = Path(__file__).resolve().parents[1] / "evals" / "fixtures" / "recall_telemetry_queries_2026-09-29.json"


def _long_query() -> str:
    rows = json.loads(FIXTURE.read_text(encoding="utf-8"))["rows"]
    longest = max((r["query"] or "" for r in rows), key=len)
    assert len(longest) == 30663
    return longest


# ── helpers ──────────────────────────────────────────────────────────────────


class _Spy:
    def __init__(self, result: Any = None, *, sleep: float = 0.0, raises: Exception | None = None):
        self.calls: List[Dict[str, Any]] = []
        self.result = [] if result is None else result
        self.sleep = sleep
        self.raises = raises

    async def __call__(self, *args, **kwargs):
        self.calls.append({"args": args, "kwargs": kwargs})
        if self.sleep:
            await asyncio.sleep(self.sleep)
        if self.raises is not None:
            raise self.raises
        return list(self.result)


def _profile() -> Dict[str, Any]:
    return {
        "profile": "reflect.v1",
        "rdf_top_k": 4,
        "max_total_items": 8,
        "max_per_source": 4,
        "render_budget_tokens": 256,
        "enable_anchor_candidates": True,
        "enable_query_expansion": True,
    }


def _bus_fragment() -> Dict[str, Any]:
    return {
        "id": "bus-anomaly-1",
        "source": "bus_synaptic_anomaly",
        "source_ref": "falkordb",
        "text": "Bus anomaly: orion:exec:request:RecallService latency spike",
        "ts": time.time(),
        "score": 0.6,
        "tags": ["bus_synaptic"],
    }


@pytest.fixture
def wired(monkeypatch):
    """Every backend enabled and replaced by a spy; no network anywhere."""
    s = worker.settings
    monkeypatch.setattr(s, "RECALL_BUS_SYNAPTIC_ANOMALY_IN_CHAT", True)
    monkeypatch.setattr(s, "RECALL_FALKOR_IN_CHAT", True)
    monkeypatch.setattr(s, "RECALL_FALKOR_NEIGHBORHOOD_IN_CHAT", True)
    monkeypatch.setattr(s, "RECALL_ENABLE_SQL_CHAT", True)
    monkeypatch.setattr(s, "RECALL_ENABLE_SQL_TIMELINE", True)
    monkeypatch.setattr(s, "RECALL_ENTITY_RELATEDNESS_BOOST_ENABLED", False)
    monkeypatch.setattr(s, "RECALL_INTENT_ROUTING_ENABLED", False)
    monkeypatch.setattr(s, "RECALL_MAX_SUB_QUERIES", 4)
    monkeypatch.setattr(s, "RECALL_MAX_QUERY_CHARS", 600)
    monkeypatch.setattr(s, "RECALL_DEADLINE_MS_DEFAULT", 60000)
    monkeypatch.setattr(worker, "get_profile", lambda _name: _profile())

    spies = {
        "bus_synaptic_anomaly": _Spy([_bus_fragment()]),
        "falkor_chat": _Spy(),
        "sql_chat_pairs": _Spy(),
        "sql_chat_msgs": _Spy(),
        "sql_timeline_recent": _Spy(),
        "sql_timeline_related": _Spy(),
        "falkor_neighborhood": _Spy(),
        "exact": _Spy(),
    }
    monkeypatch.setattr(worker, "fetch_bus_synaptic_anomaly_fragments", spies["bus_synaptic_anomaly"])
    monkeypatch.setattr(worker, "fetch_falkor_chatturn_fragments", spies["falkor_chat"])
    monkeypatch.setattr(worker, "fetch_chat_history_pairs", spies["sql_chat_pairs"])
    monkeypatch.setattr(worker, "fetch_chat_messages", spies["sql_chat_msgs"])
    monkeypatch.setattr(worker, "fetch_recent_fragments", spies["sql_timeline_recent"])
    monkeypatch.setattr(worker, "fetch_related_by_entities", spies["sql_timeline_related"])
    monkeypatch.setattr(worker, "fetch_falkor_neighborhood_fragments", spies["falkor_neighborhood"])
    monkeypatch.setattr(worker, "fetch_exact_fragments", spies["exact"])

    async def _no_ts(*_a, **_k):
        return {}

    monkeypatch.setattr(worker, "fetch_chat_turn_timestamps", _no_ts)

    import app.recall_v2 as recall_v2

    async def _shadow(q, *, profile=None):
        return MemoryBundleV1(), {}

    monkeypatch.setattr(recall_v2, "run_recall_v2_shadow", _shadow)
    return spies


FEEDS = ("bus_synaptic_anomaly", "falkor_chat", "sql_chat_pairs", "sql_chat_msgs", "sql_timeline_recent", "sql_timeline_related")


# ── 1. bounded expansion ─────────────────────────────────────────────────────


def test_long_query_intake_and_signals_are_bounded_without_stopwords(monkeypatch) -> None:
    monkeypatch.setattr(worker.settings, "RECALL_MAX_QUERY_CHARS", 600)
    monkeypatch.setattr(worker.settings, "RECALL_MAX_SUB_QUERIES", 4)
    text = _long_query()
    q = RecallQueryV1(fragment=text, verb="stance_react", profile="chat.continuity.v1")

    intake = worker._intake_query(q, profile_name="chat.continuity.v1")
    assert intake["source"] == "condensed"
    assert 0 < len(intake["search_text"]) <= 600

    for source_text in (intake["search_text"], text):
        signals = worker._expand_query(source_text, verb="stance_react", intent=None, enable=True)
        assert len(signals) <= 4 + 2
        entity_signals = signals[2:]
        for sig in entity_signals:
            assert sig.lower() not in worker._EXPANSION_STOPWORDS
            assert sig.split()[0].lower() not in worker._EXPANSION_STOPWORDS


def test_uncapped_zero_restores_old_fan_out(monkeypatch) -> None:
    monkeypatch.setattr(worker.settings, "RECALL_MAX_SUB_QUERIES", 0)
    assert worker._max_sub_queries() == 0  # a configured 0 is not eaten by a default
    signals = worker._expand_query(_long_query(), verb="stance_react", intent=None, enable=True)
    # The live slow row had 268 (fragment + verb + 266 entities).
    assert len(signals) == 268


def test_ranked_entities_are_deterministic_and_specific_first() -> None:
    text = "The Orion team moved gpu1 onto the P4 card. It said Nvidia Tesla and settings.py matter."
    first = worker._ranked_entities(text, limit=4)
    assert first == worker._ranked_entities(text, limit=4)
    assert first[0] == "settings.py"  # identifier-shaped beats plain words
    assert "The" not in first and "It" not in first
    assert "Nvidia Tesla" in first  # multi-word proper name beats single words


def test_extract_entities_is_first_appearance_ordered() -> None:
    assert worker._extract_entities("Zeta then Alpha then Zeta") == ["Zeta", "Alpha"]


def test_condense_is_deterministic_and_capped() -> None:
    text = _long_query()
    a = worker._condense_query(text, max_chars=600)
    assert a == worker._condense_query(text, max_chars=600)
    assert 0 < len(a) <= 600


def test_short_fragment_is_searched_as_is() -> None:
    q = RecallQueryV1(fragment="What can't you do right now?", verb="stance_react")
    intake = worker._intake_query(q, profile_name="reflect.v1")
    assert intake == {**intake, "source": "fragment", "search_text": "What can't you do right now?"}


# ── Phase 2 intake: caller's retrieval_query wins ────────────────────────────


def test_retrieval_query_is_searched_instead_of_fragment(wired) -> None:
    q = RecallQueryV1(
        fragment=_long_query(),
        retrieval_query="What does sentience mean here",
        verb="stance_react",
        profile="reflect.v1",
    )
    bundle, decision = asyncio.run(worker.process_recall(q, corr_id="c-caller"))
    assert decision.retrieval_query_source == "caller"
    assert decision.query == "What does sentience mean here"
    assert decision.query_chars == len("What does sentience mean here")
    first_retriever_call = wired["falkor_neighborhood"].calls[0]["kwargs"]
    assert first_retriever_call["query_text"] == "What does sentience mean here"


# ── 2. feeds once per recall ─────────────────────────────────────────────────


def test_each_context_feed_runs_once_for_n_sub_queries(wired) -> None:
    q = RecallQueryV1(fragment=_long_query(), verb="stance_react", profile="reflect.v1")
    bundle, decision = asyncio.run(worker.process_recall(q, corr_id="c-feeds"))

    n = decision.sub_query_count
    assert n is not None and 2 <= n <= 6
    for feed in FEEDS:
        assert len(wired[feed].calls) == 1, feed
    # Retrievers run once per sub-query.
    assert len(wired["falkor_neighborhood"].calls) == n
    # related_by_entities gets the bounded list, not every capitalized word.
    related_entities = wired["sql_timeline_related"].calls[0]["args"][0]
    assert len(related_entities) <= 4
    # falkor_chat gets the same window the post-filter uses, pushed into Cypher.
    assert wired["falkor_chat"].calls[0]["kwargs"]["since_minutes"] == int(worker.settings.RECALL_SQL_SINCE_MINUTES)
    assert decision.backend_counts["bus_synaptic_anomaly"] == 1


def test_single_signal_query_backends_is_unchanged(wired) -> None:
    """Direct _query_backends call (defaults): every backend once, as before."""
    cands, counts = asyncio.run(
        worker._query_backends(
            "gpu1 lane", _profile(), session_id=None, node_id=None, entities=["gpu1"], include_cards=True
        )
    )
    for feed in FEEDS:
        assert len(wired[feed].calls) == 1, feed
    assert len(wired["falkor_neighborhood"].calls) == 1
    assert "since_minutes" not in wired["falkor_chat"].calls[0]["kwargs"]
    assert counts["vector"] == 0 and counts["graph_compression"] == 0
    assert [c["id"] for c in cands] == ["bus-anomaly-1"]


# ── 3. deadline: partial results, deadline_hit ───────────────────────────────


def test_hanging_backend_returns_partial_results_within_deadline(wired) -> None:
    wired["falkor_neighborhood"].sleep = 10.0
    q = RecallQueryV1(fragment="Is gpu1 healthy?", verb="stance_react", profile="reflect.v1", deadline_ms=500)
    started = time.perf_counter()
    bundle, decision = asyncio.run(worker.process_recall(q, corr_id="c-deadline"))
    elapsed = time.perf_counter() - started
    assert elapsed < 0.5 + 0.5  # budget is 80% of 500ms, plus slack for fusion
    assert decision.deadline_hit is True
    # The feeds finished before the deadline and are kept.
    assert decision.backend_counts.get("bus_synaptic_anomaly") == 1
    assert decision.candidates_fetched and decision.candidates_fetched >= 1
    assert "bus-anomaly-1" in decision.selected_ids


def test_no_deadline_hit_when_everything_finishes(wired) -> None:
    q = RecallQueryV1(fragment="Is gpu1 healthy?", verb="stance_react", deadline_ms=5000)
    _bundle, decision = asyncio.run(worker.process_recall(q, corr_id="c-fast"))
    assert decision.deadline_hit is False


def test_one_failing_backend_does_not_kill_the_others(wired) -> None:
    wired["falkor_neighborhood"].raises = RuntimeError("falkor down")
    q = RecallQueryV1(fragment="Is gpu1 healthy?", verb="stance_react")
    _bundle, decision = asyncio.run(worker.process_recall(q, corr_id="c-fail"))
    assert decision.backend_counts.get("bus_synaptic_anomaly") == 1
    assert decision.deadline_hit is False


def test_deadline_budget_prefers_caller_at_80_percent(monkeypatch) -> None:
    monkeypatch.setattr(worker.settings, "RECALL_DEADLINE_MS_DEFAULT", 60000)
    assert worker._recall_deadline_budget_ms(RecallQueryV1(fragment="x", deadline_ms=90000)) == 72000
    assert worker._recall_deadline_budget_ms(RecallQueryV1(fragment="x")) == 60000
    monkeypatch.setattr(worker.settings, "RECALL_DEADLINE_MS_DEFAULT", 0)
    assert worker._recall_deadline_budget_ms(RecallQueryV1(fragment="x")) == 0


# ── 4. the two dead regexes, no monkeypatch ──────────────────────────────────


def test_anchor_tokens_live() -> None:
    assert worker._anchor_tokens("p4 v100 gpu1") == ["p4", "v100", "gpu1"]


def test_anchor_tokens_ignore_uuid_segments_and_hex_ids() -> None:
    text = "parent_run_id 1765808d-3a64-4be2-be03-9643f3a302bd trace cb4dd9417c8d4020 on gpu1"
    assert worker._anchor_tokens(text) == ["gpu1"]


def test_memory_browse_regex_live(monkeypatch) -> None:
    monkeypatch.setattr(worker.settings, "RECALL_BROWSE_SHORTCUT_ENABLED", True)
    assert worker._is_memory_browse("show recent memories") is True
    assert worker._is_memory_browse("list my memories") is True
    assert worker._is_memory_browse("what is the weather") is False
    # A long prompt that merely mentions "recall ... context" is not a browse request.
    assert worker._is_memory_browse(_long_query()) is False


@pytest.mark.parametrize(
    "question",
    [
        "Can you show me the context around the gpu1 crash?",
        "Recall the context of our conversation about Falkor latency",
        "list the recent errors from the scheduler",
    ],
)
def test_memory_browse_does_not_hijack_ordinary_questions(monkeypatch, question) -> None:
    """Review, PR #2416: these took the recent-only path and skipped retrieval."""
    monkeypatch.setattr(worker.settings, "RECALL_BROWSE_SHORTCUT_ENABLED", True)
    assert worker._is_memory_browse(question) is False


def test_memory_browse_off_by_default() -> None:
    assert worker.settings.RECALL_BROWSE_SHORTCUT_ENABLED is False
    assert worker._is_memory_browse("show recent memories") is False


def test_browse_flag_off_keeps_full_retrieval(wired) -> None:
    q = RecallQueryV1(fragment="show recent memories", verb="stance_react")
    _bundle, decision = asyncio.run(worker.process_recall(q, corr_id="c-browse-off"))
    assert decision.sub_query_count and decision.sub_query_count >= 1
    assert len(wired["falkor_neighborhood"].calls) >= 1


def test_anchor_rail_reaches_exact_fetch_without_monkeypatching_tokens(wired) -> None:
    asyncio.run(
        worker._fetch_anchor_candidates(
            query_text="did gpu1 and p4 recover",
            session_id=None,
            node_id=None,
            profile=_profile(),
        )
    )
    assert wired["exact"].calls[0]["kwargs"]["tokens"] == ["gpu1", "p4"]


# ── context_only mode ────────────────────────────────────────────────────────


def test_context_only_runs_feeds_and_no_retrievers(wired) -> None:
    q = RecallQueryV1(fragment="", verb="reverie_narrate", mode="context_only")
    _bundle, decision = asyncio.run(worker.process_recall(q, corr_id="c-ctx"))
    assert wired["falkor_neighborhood"].calls == []
    assert wired["exact"].calls == []  # no anchor rail
    for feed in ("bus_synaptic_anomaly", "falkor_chat", "sql_chat_pairs", "sql_timeline_recent"):
        assert len(wired[feed].calls) == 1, feed
    assert wired["falkor_chat"].calls[0]["kwargs"]["allow_empty_query"] is True
    assert decision.sub_query_count == 0


# ── rollback lever end to end ────────────────────────────────────────────────


def test_max_sub_queries_zero_is_uncapped_end_to_end(wired, monkeypatch) -> None:
    monkeypatch.setattr(worker.settings, "RECALL_MAX_SUB_QUERIES", 0)
    monkeypatch.setattr(worker.settings, "RECALL_MAX_QUERY_CHARS", 0)
    q = RecallQueryV1(fragment=_long_query(), verb="stance_react")
    _bundle, decision = asyncio.run(worker.process_recall(q, corr_id="c-uncapped"))
    assert decision.sub_query_count == 268
    assert decision.retrieval_query_source == "fragment"
    # Feeds still run once: the split is structural, not a knob.
    assert len(wired["bus_synaptic_anomaly"].calls) == 1


# ── telemetry fields ─────────────────────────────────────────────────────────


def test_decision_carries_stage_timings_and_counts(wired) -> None:
    q = RecallQueryV1(fragment=_long_query(), verb="stance_react")
    bundle, decision = asyncio.run(worker.process_recall(q, corr_id="c-telemetry"))
    for stage in ("intake", "feeds", "retrievers", "windowing", "boost", "fusion", "total"):
        assert stage in decision.timings_ms, stage
    assert decision.latency_ms == decision.timings_ms["total"] == bundle.stats.latency_ms
    assert decision.retrieval_query_source == "condensed"
    assert decision.query_chars == len(decision.query) <= 600
    assert decision.candidates_fetched is not None and decision.candidates_kept is not None
    assert decision.candidates_kept <= decision.candidates_fetched


# ── falkor_chat window pushed into Cypher ────────────────────────────────────


class _FakeFalkor:
    def __init__(self):
        self.calls: List[tuple] = []

    def graph_query(self, cypher, params=None):
        self.calls.append((cypher, dict(params or {})))
        return []


def test_falkor_chat_pushes_window_into_cypher(monkeypatch) -> None:
    fake = _FakeFalkor()
    monkeypatch.setattr(falkor_chat_adapter, "get_recall_falkor_client", lambda: fake)
    asyncio.run(
        falkor_chat_adapter.fetch_falkor_chatturn_fragments(query_text="x", session_id=None, since_minutes=180)
    )
    cypher, params = fake.calls[0]
    assert "WHERE t.ts >= $cutoff" in cypher
    # Whole seconds, no fraction/offset: 2026-09-29T04:11:00
    assert len(params["cutoff"]) == 19


def test_falkor_chat_without_window_is_unchanged(monkeypatch) -> None:
    fake = _FakeFalkor()
    monkeypatch.setattr(falkor_chat_adapter, "get_recall_falkor_client", lambda: fake)
    asyncio.run(falkor_chat_adapter.fetch_falkor_chatturn_fragments(query_text="x", session_id=None))
    cypher, params = fake.calls[0]
    assert "WHERE" not in cypher and "cutoff" not in params


def test_falkor_chat_empty_query_only_with_allow_flag(monkeypatch) -> None:
    fake = _FakeFalkor()
    monkeypatch.setattr(falkor_chat_adapter, "get_recall_falkor_client", lambda: fake)
    asyncio.run(falkor_chat_adapter.fetch_falkor_chatturn_fragments(query_text="", session_id=None))
    assert fake.calls == []
    asyncio.run(
        falkor_chat_adapter.fetch_falkor_chatturn_fragments(query_text="", session_id=None, allow_empty_query=True)
    )
    assert len(fake.calls) == 1


def test_falkor_cutoff_orders_like_stored_timestamps() -> None:
    from datetime import datetime, timezone

    now = datetime(2026, 9, 29, 7, 11, 0, 120601, tzinfo=timezone.utc)
    cutoff = falkor_chat_adapter._falkor_ts_cutoff(180, now=now)
    assert cutoff == "2026-09-29T04:11:00"
    # Both stored shapes isoformat() produces, inside the window:
    assert "2026-09-29T04:11:00.120602+00:00" >= cutoff
    assert "2026-09-29T04:11:00+00:00" >= cutoff  # microsecond == 0: no fraction
    assert "2026-09-29T04:11:01+00:00" >= cutoff
    # ...and outside it, in both shapes:
    assert "2026-09-29T04:10:59.999999+00:00" < cutoff
    assert "2026-09-29T04:10:59+00:00" < cutoff


def test_falkor_cutoff_with_whole_second_now() -> None:
    from datetime import datetime, timezone

    cutoff = falkor_chat_adapter._falkor_ts_cutoff(0, now=datetime(2026, 9, 29, 4, 11, 0, 0, tzinfo=timezone.utc))
    assert "2026-09-29T04:11:00+00:00" >= cutoff
    assert "2026-09-29T04:10:59.999999+00:00" < cutoff


# ── fusion tokenizes the query once ──────────────────────────────────────────


def test_fusion_tokenizes_query_once_per_query(monkeypatch) -> None:
    from app import fusion

    fusion._query_token_set.cache_clear()
    fusion._query_rare_tokens_lower.cache_clear()
    cands = [
        {"id": f"c{i}", "source": "sql_chat", "text": f"gpu1 lane report number {i}", "score": 0.5, "tags": []}
        for i in range(50)
    ]
    fusion.fuse_candidates(candidates=cands, profile=_profile(), query_text="gpu1 lane health", latency_ms=0)
    info = fusion._query_token_set.cache_info()
    assert info.misses == 1 and info.hits >= 49


def test_condense_puts_the_standing_question_first() -> None:
    condensed = worker._condense_query(_long_query(), max_chars=600)
    assert condensed.startswith("What does sentience mean here, and do the attributes I have found track it")


def test_condense_without_questions_uses_informative_score() -> None:
    text = "hi. The gpu1 lane stalled during the p4 migration window yesterday. ok"
    # Social filler dropped; the one substantive clause is over budget, so it is hard-cut.
    assert worker._condense_query(text, max_chars=60) == "The gpu1 lane stalled during the p4 migration window yesterd"
    assert worker._condense_query(text, max_chars=200) == "The gpu1 lane stalled during the p4 migration window yesterday"



# ── Code review (PR #2416) regressions ───────────────────────────────────────


def _blocking(seconds: float, result):
    """SYNC stub that blocks its thread with time.sleep. An async spy can't
    catch loop-blocking; this one freezes the event loop if called on it."""

    calls: List[Dict[str, Any]] = []

    def _fn(*args, **kwargs):
        import threading

        calls.append({**kwargs, "_on_loop_thread": threading.current_thread() is threading.main_thread()})
        time.sleep(seconds)
        return list(result)

    _fn.calls = calls  # type: ignore[attr-defined]
    return _fn


async def _run_with_ticker(coro):
    """Run ``coro`` while a ticker counts event-loop turns. Returns
    (result, elapsed_s, ticks). A blocked loop shows up as ~0 ticks."""
    ticks = 0
    stop = False

    async def _tick():
        nonlocal ticks
        while not stop:
            await asyncio.sleep(0.01)
            ticks += 1

    t = asyncio.ensure_future(_tick())
    started = time.perf_counter()
    try:
        result = await coro
    finally:
        stop = True
        await t
    return result, time.perf_counter() - started, ticks


def _rdf_anchor_profile() -> Dict[str, Any]:
    return {**_profile(), "enable_rdf": True}


def test_anchor_rdf_exact_runs_off_the_event_loop_and_keeps_sql_partial(wired, monkeypatch) -> None:
    """Review finding 2 + anchor-sink nit: the sync RDF exact-match call
    (requests.post) used to run on the event loop inside the concurrent fetch.
    With a 1.5s blocking stub and a 500ms deadline, recall must return on time
    with the SQL half of the anchor rail kept."""
    monkeypatch.setattr(worker, "get_profile", lambda _n: _rdf_anchor_profile())
    monkeypatch.setattr(worker.settings, "RECALL_RDF_ENDPOINT_URL", "http://rdf.invalid")
    blocking = _blocking(1.5, [])
    monkeypatch.setattr(worker, "fetch_rdf_chatturn_exact_matches", blocking)
    exact_item = type(
        "Row", (), {"id": "anchor-sql-1", "source_ref": "chat_history_log", "text": "gpu1 lane recovered", "ts": time.time(), "tags": []}
    )()
    wired["exact"].result = [exact_item]

    q = RecallQueryV1(fragment="did gpu1 recover", verb="stance_react", deadline_ms=500)
    (bundle, decision), elapsed, ticks = asyncio.run(
        _run_with_ticker(worker.process_recall(q, corr_id="c-anchor-block"))
    )
    assert blocking.calls, "rdf exact stub was not reached"
    assert not any(c["_on_loop_thread"] for c in blocking.calls)
    assert elapsed < 1.2
    assert ticks >= 20  # loop kept turning while the stub slept
    assert decision.deadline_hit is True
    assert decision.backend_counts.get("sql_timeline_anchor") == 1
    assert "anchor-sql-1" in decision.selected_ids


def test_anchor_rail_shares_the_fetch_semaphore(wired, monkeypatch) -> None:
    """Review finding 5: with RECALL_FETCH_CONCURRENCY=1, the anchor rail and
    the backend units never overlap."""
    monkeypatch.setattr(worker.settings, "RECALL_FETCH_CONCURRENCY", 1)
    in_flight = 0
    peak = 0

    def _tracked(spy):
        async def _fn(*a, **k):
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)
            try:
                await asyncio.sleep(0.01)
                return await spy(*a, **k)
            finally:
                in_flight -= 1

        return _fn

    for name, attr in (
        ("exact", "fetch_exact_fragments"),
        ("falkor_neighborhood", "fetch_falkor_neighborhood_fragments"),
        ("bus_synaptic_anomaly", "fetch_bus_synaptic_anomaly_fragments"),
        ("falkor_chat", "fetch_falkor_chatturn_fragments"),
    ):
        monkeypatch.setattr(worker, attr, _tracked(wired[name]))
    q = RecallQueryV1(fragment="did gpu1 and p4 recover on Tesla", verb="stance_react")
    asyncio.run(worker.process_recall(q, corr_id="c-sem"))
    assert wired["exact"].calls, "anchor rail did not run"
    assert peak == 1


def test_fetch_concurrency_default_is_four(monkeypatch) -> None:
    assert worker.settings.RECALL_FETCH_CONCURRENCY == 4
    assert worker._fetch_concurrency() == 4


def test_shadow_compare_not_triggered_by_anchor_tokens_alone(wired, monkeypatch) -> None:
    """Review finding 3: main's triggers only (empty or vector-topped bundle)."""
    import app.recall_v2 as recall_v2

    calls: List[Any] = []

    async def _shadow(q, *, profile=None):
        calls.append(q)
        return MemoryBundleV1(), {}

    monkeypatch.setattr(recall_v2, "run_recall_v2_shadow", _shadow)
    q = RecallQueryV1(fragment="did gpu1 recover", verb="stance_react")
    bundle, _decision = asyncio.run(worker.process_recall(q, corr_id="c-shadow-anchor"))
    assert bundle.items  # bus anomaly feed keeps it non-empty
    assert worker._anchor_tokens("did gpu1 recover") == ["gpu1"]
    assert calls == []

    wired["bus_synaptic_anomaly"].result = []
    asyncio.run(worker.process_recall(q, corr_id="c-shadow-empty"))
    assert len(calls) == 1  # empty bundle still triggers it, as on main


def test_v2_shadow_sync_calls_run_off_the_event_loop(monkeypatch) -> None:
    """Review finding 3: the shadow's sync RDF / pageindex calls go to threads."""
    import app.recall_v2 as recall_v2

    async def _empty(*a, **k):
        return []

    blocking_rdf = _blocking(0.4, [])
    blocking_exact = _blocking(0.4, [])
    blocking_pageindex = _blocking(0.4, [])
    monkeypatch.setattr(recall_v2, "fetch_rdf_fragments", blocking_rdf)
    monkeypatch.setattr(recall_v2, "fetch_rdf_chatturn_exact_matches", blocking_exact)
    monkeypatch.setattr(recall_v2, "_pageindex_candidates", lambda plan, top_k=8: blocking_pageindex(top_k=top_k))
    monkeypatch.setattr(recall_v2, "fetch_exact_fragments", _empty)
    monkeypatch.setattr(recall_v2, "fetch_recent_fragments", _empty)

    q = RecallQueryV1(fragment="find exact anchor COMMIT123 in memory", profile="reflect.v1")
    _result, elapsed, ticks = asyncio.run(_run_with_ticker(recall_v2.run_recall_v2_shadow(q, profile=_profile())))
    assert blocking_rdf.calls and blocking_exact.calls and blocking_pageindex.calls
    for stub in (blocking_rdf, blocking_exact, blocking_pageindex):
        assert not any(c["_on_loop_thread"] for c in stub.calls)
    assert elapsed >= 1.0
    # ~1.2s of sync sleeping; on-loop it would be ~0 ticks.
    assert ticks >= 50


def test_boost_gets_same_entity_count_as_fallback(wired, monkeypatch) -> None:
    """Nit: process_recall and the boost's own fallback pass max(K,3) ranked entities."""
    monkeypatch.setattr(worker.settings, "RECALL_ENTITY_RELATEDNESS_BOOST_ENABLED", True)
    monkeypatch.setattr(worker.settings, "RECALL_MAX_SUB_QUERIES", 1)
    seen: List[List[str]] = []

    async def _boost(*, query_text, candidates, query_entities=None):
        seen.append(list(query_entities or []))
        return {}, []

    monkeypatch.setattr(worker, "_compute_entity_relatedness_boost_map", _boost)
    text = "Check settings.py, Nvidia Tesla, gpu1 and Falkor Graph after the Atlas move"
    asyncio.run(worker.process_recall(RecallQueryV1(fragment=text, verb="stance_react"), corr_id="c-boost"))
    assert seen and len(seen[0]) == 3
    assert seen[0] == worker._boost_query_entities(text)


def test_condense_never_empty_for_non_empty_input() -> None:
    ws = " " * 700
    assert worker._condense_query(ws, max_chars=600) != ""
    punct = "." * 700
    assert worker._condense_query(punct, max_chars=600) != ""


def test_intake_skips_whitespace_only_fragment(monkeypatch) -> None:
    monkeypatch.setattr(worker.settings, "RECALL_MAX_QUERY_CHARS", 600)
    intake = worker._intake_query(RecallQueryV1(fragment=" " * 700), profile_name="reflect.v1")
    assert intake["search_text"] == "" and intake["source"] == "fragment"


def test_condense_hard_cut_counts_toward_budget() -> None:
    long_clause = "gpu1 " * 200  # one 1000-char clause, no terminators
    out = worker._condense_query(long_clause + ". short tail clause here", max_chars=100)
    assert len(out) <= 100
    assert "short tail" not in out
