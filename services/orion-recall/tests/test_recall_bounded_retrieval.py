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


def test_memory_browse_regex_live() -> None:
    assert worker._is_memory_browse("show recent memories") is True
    assert worker._is_memory_browse("what is the weather") is False
    # A long prompt that merely mentions "recall ... context" is not a browse request.
    assert worker._is_memory_browse(_long_query()) is False


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
    # Same shape the writer stores: 2026-09-29T04:11:00.120601+00:00
    assert len(params["cutoff"]) == 32 and params["cutoff"].endswith("+00:00")


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
    assert cutoff == "2026-09-29T04:11:00.120601+00:00"
    assert "2026-09-29T04:11:00.120602+00:00" >= cutoff
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
