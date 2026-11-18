"""Lived-memory recall step 2 (issue #3534) and the degraded fallback (issue #3573).

Query rewrite, link-following over episode_memory_referent, and exact-word ranking when the
reranker or embedder is down. Made-up records only; no DB, bus or model (fakes only), except
one test that points the real HTTP reranker call at a port nothing listens on.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

import pytest

import app.lived_memory as lm
from app.lived_links import ReferentGraph, usable_name
from app.lived_memory import LivedMemoryConfig, LivedMemoryRecall, rerank_query, rrf, tokenize
from app.lived_rewrite import ContextTurn, build_prompt, fetch_context, parse_queries
from orion.introspect.semantic_index import SearchConfig

NOW = datetime(2026, 11, 16, 12, 0, tzinfo=timezone.utc)


def _row(rid: str, text: str, **kw: Any) -> Dict[str, Any]:
    at = kw.get("at", NOW - timedelta(days=2))
    return {"record_id": rid, "purpose": "happened", "text": text, "occurred_at": at, "importance": 0.8,
            "is_correction": False, "confirmed": False, "label": None, "created_at": at}


class Pool:
    """asyncpg-shaped fake: the serving queries, the referent graph and chat_history_log."""

    def __init__(self, live: Dict[str, List[Dict[str, Any]]], *, edges=(), names=(), chat=()) -> None:
        self.live = live
        self.edges = list(edges)
        self.names = list(names)
        self.chat = list(chat)
        self.fetches: List[str] = []

    async def fetch(self, sql: str, *args: Any) -> List[Dict[str, Any]]:
        self.fetches.append(sql)
        if "FROM episode_memory_referent" in sql:
            return [{"memory_id": m, "referent_key": r} for m, r in self.edges]
        if "FROM referent_alias" in sql:
            return [{"alias": a, "referent_key": r} for a, r in self.names]
        if "FROM chat_history_log" in sql:
            session, before, after, limit = args
            rows = [c for c in self.chat if c["session_id"] == session and after <= c["created_at"] < before]
            rows.sort(key=lambda c: c["created_at"], reverse=True)
            return rows[:limit]
        kind = ("episode_memory" if "FROM episode_memory" in sql else
                "stance" if "FROM memory_crystallizations" in sql else "person_fact")
        rows = list(self.live.get(kind, []))
        if args:
            rows = [r for r in rows if r["record_id"] in set(args[0])]
        return rows


def _cfg(**kw: Any) -> LivedMemoryConfig:
    base = dict(search=SearchConfig("http://chroma", "http://embed", "orion_lived_memory", 0.0),
                rerank_url="http://rerank", min_fit=0.5, max_lines=4, timeout_sec=2.0, min_query_tokens=5,
                rewrite_timeout_sec=0.5, link_top_k=6)
    base.update(kw)
    return LivedMemoryConfig(**base)


def overlap_reranker(query: str, docs: List[str]):
    q = set(tokenize(query))

    async def go():
        return [(-4.0 + 3.0 * len(q & set(tokenize(d)))) for d in docs]

    return go()


def fixed_rewriter(queries: List[str], seen: List[Any] | None = None):
    async def rw(message, context):
        if seen is not None:
            seen.append((message, list(context)))
        return list(queries)

    return rw


@pytest.fixture
def no_dense(monkeypatch):
    async def embed(*_a, **_k):
        return [0.0], "m"

    async def nearest(*_a, **_k):
        return []

    monkeypatch.setattr(lm, "embed", embed)
    monkeypatch.setattr(lm, "nearest", nearest)


GAME = _row("g1", "Juniper told me she reached the final dungeon of the old castle adventure game.")
OFFICE = _row("o1", "The office was loud all evening.")


# --- rewrite: parsing and prompt -----------------------------------------------------------

def test_parse_queries_reads_lines_and_strips_bullets():
    text = "1. castle adventure game final dungeon\n- beat the castle game\n* castle game save states\nextra line here"
    assert parse_queries(text) == ["castle adventure game final dungeon", "beat the castle game",
                                   "castle game save states"]


def test_parse_queries_none_means_no_search():
    assert parse_queries("NONE") == []
    assert parse_queries("none.") == []
    assert parse_queries("") == []


def test_parse_queries_drops_one_word_fragments_and_duplicates():
    assert parse_queries("game\ncastle game ending\nCastle game ending") == ["castle game ending"]


def test_build_prompt_carries_context_oldest_first_and_the_message():
    prompt = build_prompt("okay, I beat it!", [ContextTurn("off to the last dungeon", "Good luck!")])
    assert prompt.index("User: off to the last dungeon") < prompt.index("Orion: Good luck!") < prompt.index("okay, I beat it!")
    assert "NONE" in prompt


def test_fetch_context_reads_this_session_only_and_skips_the_active_turn():
    t0 = datetime(2026, 11, 16, 11, 0)
    chat = [
        {"session_id": "s1", "created_at": t0, "prompt": "off to the last dungeon", "response": "Good luck!",
         "correlation_id": "c-old"},
        {"session_id": "s1", "created_at": t0 + timedelta(minutes=30), "prompt": "okay, I beat it!",
         "response": "", "correlation_id": "c-now"},
        {"session_id": "s2", "created_at": t0, "prompt": "other conversation", "response": "x",
         "correlation_id": "c-x"},
        {"session_id": "s1", "created_at": t0 - timedelta(hours=9), "prompt": "too old", "response": "x",
         "correlation_id": "c-y"},
    ]
    pool = Pool({}, chat=chat)
    turns = asyncio.run(fetch_context(pool, "s1", before=NOW, exclude_text="okay, I beat it!",
                                      exclude_ids=["c-now"]))
    assert [t.user for t in turns] == ["off to the last dungeon"]
    assert asyncio.run(fetch_context(pool, None, before=NOW)) == []


# --- rewrite: in the recall --------------------------------------------------------------

def test_an_implicit_message_finds_its_memory_through_the_rewrite(no_dense):
    t0 = datetime(2026, 11, 16, 11, 30)
    pool = Pool({"episode_memory": [GAME, OFFICE]},
                chat=[{"session_id": "s1", "created_at": t0, "prompt": "off to the castle game's last dungeon",
                       "response": "Good luck!", "correlation_id": "c-old"}])
    seen: List[Any] = []
    reader = LivedMemoryRecall(pool, _cfg(), reranker=overlap_reranker,
                               rewriter=fixed_rewriter(["castle adventure game final dungeon"], seen),
                               clock=lambda: NOW)
    msg = "okay, finally beat it after tries tonight, whew!"
    out = asyncio.run(reader.recall(msg, session_id="s1"))
    assert [s.rec.record_id for s in out.chosen] == ["g1"]
    assert out.debug["rewrite"]["outcome"] == "ok" and out.backend == "hybrid"
    assert seen[0][1][0].user == "off to the castle game's last dungeon"
    # Without the rewrite the message's own words fit nothing.
    plain = LivedMemoryRecall(pool, _cfg(), reranker=overlap_reranker, clock=lambda: NOW)
    assert asyncio.run(plain.recall(msg)).lines == []


def test_short_messages_are_neither_rewritten_nor_searched(no_dense):
    calls: List[Any] = []
    reader = LivedMemoryRecall(Pool({"episode_memory": [GAME]}), _cfg(), reranker=overlap_reranker,
                               rewriter=fixed_rewriter(["castle adventure game final dungeon"], calls), clock=lambda: NOW)
    out = asyncio.run(reader.recall("okay, I beat it!"))
    assert out.lines == [] and out.rendered == "" and out.debug["skipped"] == "few_content_words"
    assert calls == []


def test_small_talk_stays_empty_when_the_rewrite_says_none(no_dense):
    reader = LivedMemoryRecall(Pool({"episode_memory": [GAME]}), _cfg(), reranker=overlap_reranker,
                               rewriter=fixed_rewriter([]), clock=lambda: NOW)
    out = asyncio.run(reader.recall("sounds lovely, enjoy your evening and sleep well tonight"))
    assert out.lines == [] and out.rendered == "" and out.debug["rewrite"]["outcome"] == "none"


def test_a_slow_rewrite_falls_back_to_the_raw_message(no_dense):
    async def slow(message, context):
        await asyncio.sleep(5)
        return ["never used here"]

    reader = LivedMemoryRecall(Pool({"episode_memory": [GAME]}), _cfg(rewrite_timeout_sec=0.05),
                               reranker=overlap_reranker, rewriter=slow, clock=lambda: NOW)
    out = asyncio.run(reader.recall("Juniper reached the final dungeon of the castle adventure game today"))
    assert out.debug["rewrite"]["outcome"] == "timeout"
    assert [s.rec.record_id for s in out.chosen] == ["g1"]
    assert out.debug["timings_ms"]["total"] < 1500


def test_a_failing_rewrite_never_fails_the_recall(no_dense):
    async def broken(message, context):
        raise RuntimeError("gateway refused")

    reader = LivedMemoryRecall(Pool({"episode_memory": [GAME]}), _cfg(), reranker=overlap_reranker,
                               rewriter=broken, clock=lambda: NOW)
    out = asyncio.run(reader.recall("Juniper reached the final dungeon of the castle adventure game today"))
    assert out.debug["rewrite"]["outcome"] == "error:RuntimeError" and out.lines


def test_a_long_message_skips_the_rewrite(no_dense):
    calls: List[Any] = []
    reader = LivedMemoryRecall(Pool({"episode_memory": [GAME]}), _cfg(), reranker=overlap_reranker,
                               rewriter=fixed_rewriter(["x y"], calls), clock=lambda: NOW)
    out = asyncio.run(reader.recall("castle adventure game final dungeon " * 60))
    assert out.debug["rewrite"]["outcome"] == "skipped_long_message" and calls == []


def test_rerank_query_appends_the_searches_and_rrf_rewards_agreement():
    assert rerank_query("okay, I beat it!", []) == "okay, I beat it!"
    assert rerank_query("okay, I beat it!", ["castle game", "final dungeon"]) == "okay, I beat it! | castle game ; final dungeon"
    fused = rrf([[("a", 1.0), ("b", 0.5)], [("b", 3.0)]])
    assert fused["b"] > fused["a"]


# --- link-following ----------------------------------------------------------------------

def test_usable_name_rejects_function_words_short_and_numeric_aliases():
    assert not usable_name("that") and not usable_name("hub") and not usable_name("2666")
    assert usable_name("castle game") and usable_name("Biscuit")


def test_graph_drops_hub_referents_and_spreads_through_shared_ones():
    edges = [("m1", "project:castle-game"), ("m2", "project:castle-game"), ("m3", "place:office")]
    edges += [(f"m{i}", "person:juniper") for i in range(1, 40)]
    g = ReferentGraph(edges, [("castle game", "project:castle-game"), ("that", "project:castle-game"),
                              ("juniper", "person:juniper")], hub_max_degree=30)
    assert "person:juniper" in g.hubs
    assert g.mentions("I love that castle game!") == {"project:castle-game"}
    assert g.mentions("hi Juniper") == set()  # a hub is not a link
    got = dict(g.spread(["episode_memory:m1"], exclude={"episode_memory:m1"}))
    assert "episode_memory:m2" in got and "episode_memory:m3" not in got
    named = dict(g.spread([], {"project:castle-game"}, exclude=set()))
    assert set(named) == {"episode_memory:m1", "episode_memory:m2"}


def test_a_linked_memory_joins_the_rerank_and_can_be_served(no_dense):
    seed = _row("s1", "Juniper reached the final dungeon of the castle adventure game.")
    sibling = _row("s2", "She said the save states rescued her whole run.")
    pool = Pool({"episode_memory": [seed, sibling, OFFICE]},
                edges=[("s1", "project:castle-game"), ("s2", "project:castle-game"), ("o1", "place:office")])

    def reranker(query, docs):
        async def go():  # the sibling fits the turn even though the searches never found it
            return [3.0 if "save states" in d else (1.0 if "final dungeon" in d else -6.0) for d in docs]
        return go()

    reader = LivedMemoryRecall(pool, _cfg(), reranker=reranker, clock=lambda: NOW)
    out = asyncio.run(reader.recall("final dungeon reached in the castle adventure game today"))
    ids = [s.rec.record_id for s in out.chosen]
    assert "s2" in ids and next(s for s in out.chosen if s.rec.record_id == "s2").link
    assert out.debug["candidates"]["links"] >= 1
    assert "link" in lm.items_for(out.chosen)[ids.index("s2")]["tags"]
    off = LivedMemoryRecall(pool, _cfg(link_top_k=0), reranker=reranker, clock=lambda: NOW)
    assert "s2" not in [s.rec.record_id for s in asyncio.run(off.recall("final dungeon reached in the castle adventure game today")).chosen]


def test_a_linked_memory_below_the_bar_is_never_served(no_dense):
    seed = _row("s1", "Juniper reached the final dungeon of the castle adventure game.")
    sibling = _row("s2", "Unrelated note that shares only a referent.")
    pool = Pool({"episode_memory": [seed, sibling]},
                edges=[("s1", "project:castle-game"), ("s2", "project:castle-game")])
    reader = LivedMemoryRecall(pool, _cfg(), reranker=overlap_reranker, clock=lambda: NOW)
    out = asyncio.run(reader.recall("final dungeon reached in the castle adventure game today"))
    assert [s.rec.record_id for s in out.chosen] == ["s1"]


def test_links_off_when_the_referent_tables_cannot_be_read(no_dense):
    class NoGraph(Pool):
        async def fetch(self, sql, *args):
            if "referent" in sql:
                raise RuntimeError("permission denied")
            return await super().fetch(sql, *args)

    reader = LivedMemoryRecall(NoGraph({"episode_memory": [GAME]}), _cfg(), reranker=overlap_reranker,
                               clock=lambda: NOW)
    out = asyncio.run(reader.recall("Juniper reached the final dungeon of the castle adventure game today"))
    assert out.lines and "links" not in out.debug


# --- degraded mode (issue #3573) ---------------------------------------------------------

def test_reranker_down_falls_back_to_exact_words_and_says_so(no_dense):
    async def down(query, docs):
        raise ConnectionError("rerank unreachable")

    reader = LivedMemoryRecall(Pool({"episode_memory": [GAME, OFFICE]}), _cfg(), reranker=down, clock=lambda: NOW)
    out = asyncio.run(reader.recall("Juniper reached the final dungeon of the castle adventure game today"))
    assert out.backend == "degraded" and out.debug["degraded"] == ["rerank_unavailable"]
    assert [s.rec.record_id for s in out.chosen] == ["g1"] and out.chosen[0].degraded
    assert out.rendered.startswith(lm.DEGRADED_HEADER)
    assert "degraded" in lm.items_for(out.chosen)[0]["tags"]


def test_reranker_too_slow_degrades_inside_the_deadline(no_dense):
    def slow(query, docs):
        async def go():
            await asyncio.sleep(10)
            return [1.0] * len(docs)
        return go()

    reader = LivedMemoryRecall(Pool({"episode_memory": [GAME]}), _cfg(timeout_sec=0.6), reranker=slow,
                               clock=lambda: NOW)
    out = asyncio.run(reader.recall("Juniper reached the final dungeon of the castle adventure game today"))
    assert out.debug["degraded"] == ["rerank_timeout"] and [s.rec.record_id for s in out.chosen] == ["g1"]
    assert out.debug["timings_ms"]["total"] < 900


def test_reranker_endpoint_down_over_real_http_degrades(no_dense):
    """No fake reranker: the real HTTP call goes to a port nothing listens on."""
    reader = LivedMemoryRecall(Pool({"episode_memory": [GAME, OFFICE]}),
                               _cfg(rerank_url="http://127.0.0.1:9/rerank"), clock=lambda: NOW)
    out = asyncio.run(reader.recall("Juniper reached the final dungeon of the castle adventure game today"))
    assert out.backend == "degraded" and out.debug["degraded"] == ["rerank_unavailable"]
    assert [s.rec.record_id for s in out.chosen] == ["g1"]


def test_embedder_down_is_labelled_degraded_but_still_reranked(monkeypatch):
    from orion.introspect.semantic_index import SearchUnavailableError

    async def embed(*_a, **_k):
        raise SearchUnavailableError("embedder unavailable: ConnectError")

    monkeypatch.setattr(lm, "embed", embed)
    reader = LivedMemoryRecall(Pool({"episode_memory": [GAME]}), _cfg(), reranker=overlap_reranker,
                               clock=lambda: NOW)
    out = asyncio.run(reader.recall("Juniper reached the final dungeon of the castle adventure game today"))
    assert out.backend == "degraded" and out.debug["degraded"] == ["embedder_unavailable"]
    assert out.lines and not out.chosen[0].degraded and out.chosen[0].fit > 0.5


def test_degraded_with_no_match_still_tells_the_prompt(no_dense):
    async def down(query, docs):
        raise ConnectionError("down")

    reader = LivedMemoryRecall(Pool({"episode_memory": [GAME]}), _cfg(), reranker=down, clock=lambda: NOW)
    # One shared content word ("adventure") is too weak to serve without a fit score.
    out = asyncio.run(reader.recall("thinking about an adventure with weather and baking bread today"))
    assert out.lines == [] and out.backend == "degraded" and out.rendered == lm.DEGRADED_EMPTY_NOTE


def test_a_link_neighbour_is_never_served_without_a_rerank(no_dense, monkeypatch):
    """The neighbour shares words with the turn, so exact words alone would serve it; it entered
    the candidates only as a link (pushed out of the searched pool), so it must not be served."""
    async def down(query, docs):
        raise ConnectionError("down")

    monkeypatch.setattr(lm, "MAX_RERANK", 2)
    monkeypatch.setattr(lm, "MAX_RERANK_STEP2", 2)
    seed = _row("s1", "Juniper reached the final dungeon of the castle adventure game.")
    other = _row("s3", "The castle adventure game has a final dungeon boss.")
    sibling = _row("s2", "Castle adventure game notes from the final week.")
    pool = Pool({"episode_memory": [seed, other, sibling]},
                edges=[("s1", "project:castle-game"), ("s2", "project:castle-game")])
    reader = LivedMemoryRecall(pool, _cfg(), reranker=down, clock=lambda: NOW)
    out = asyncio.run(reader.recall("final dungeon reached in the castle adventure game today"))
    assert out.debug["candidates"]["links"] == 1  # s2 came in as a link, not as a search hit
    assert out.backend == "degraded" and "s2" not in [s.rec.record_id for s in out.chosen]
    assert out.chosen


def test_a_hanging_embedder_degrades_inside_the_deadline(monkeypatch):
    async def embed(*_a, **_k):
        await asyncio.sleep(10)
        return [0.0], "m"

    monkeypatch.setattr(lm, "embed", embed)
    monkeypatch.setattr(lm, "RERANK_RESERVE_SEC", 0.3)
    reader = LivedMemoryRecall(Pool({"episode_memory": [GAME]}), _cfg(timeout_sec=0.8), reranker=overlap_reranker,
                               clock=lambda: NOW)
    out = asyncio.run(reader.recall("Juniper reached the final dungeon of the castle adventure game today"))
    assert out.backend == "degraded" and out.debug["degraded"] == ["embedder_timeout"]
    assert [s.rec.record_id for s in out.chosen] == ["g1"] and not out.chosen[0].degraded  # still reranked
    assert out.debug["timings_ms"]["total"] < 1000


def test_degraded_small_talk_gets_no_note(no_dense):
    async def down(query, docs):
        raise ConnectionError("down")

    reader = LivedMemoryRecall(Pool({"episode_memory": [GAME]}), _cfg(), reranker=down,
                               rewriter=fixed_rewriter([]), clock=lambda: NOW)
    out = asyncio.run(reader.recall("sounds lovely, enjoy your evening and sleep well tonight"))
    assert out.lines == [] and out.rendered == ""


def test_embedder_down_and_nothing_fits_says_meaning_search_was_down(monkeypatch):
    from orion.introspect.semantic_index import SearchUnavailableError

    async def embed(*_a, **_k):
        raise SearchUnavailableError("down")

    monkeypatch.setattr(lm, "embed", embed)
    reader = LivedMemoryRecall(Pool({"episode_memory": [GAME]}), _cfg(min_fit=0.999), reranker=overlap_reranker,
                               clock=lambda: NOW)
    out = asyncio.run(reader.recall("thinking about an adventure with weather and baking bread today"))
    assert out.lines == [] and out.rendered == lm.DEGRADED_EMPTY_NOTE_RERANKED


def test_names_in_the_rewrite_seed_the_links(no_dense):
    sibling = _row("s2", "She said the save states rescued her whole run.")
    pool = Pool({"episode_memory": [GAME, sibling]},
                edges=[("g1", "project:castle-game"), ("s2", "project:castle-game")],
                names=[("castle quest", "project:castle-game")])
    reader = LivedMemoryRecall(pool, _cfg(), reranker=overlap_reranker,
                               rewriter=fixed_rewriter(["castle quest ending"]), clock=lambda: NOW)
    out = asyncio.run(reader.recall("okay, finally beat it after tries tonight, whew!"))
    assert out.debug["links"]["seed_names"] == 1


# --- worker seam --------------------------------------------------------------------------

def test_worker_renders_the_degraded_note_and_passes_the_session(monkeypatch, no_dense):
    from app import worker
    from orion.core.contracts.recall import MemoryBundleStatsV1, MemoryBundleV1

    async def down(query, docs):
        raise ConnectionError("down")

    seen: List[Any] = []
    reader = LivedMemoryRecall(Pool({"episode_memory": [GAME]}), _cfg(), reranker=down,
                               rewriter=fixed_rewriter(["weather and bread"], seen), clock=lambda: NOW)
    worker.set_lived_memory(reader)
    try:
        bundle = MemoryBundleV1(rendered="[Recent thread]\nUser: hi", items=[], stats=MemoryBundleStatsV1())

        async def go():
            task = worker._start_lived_recall("thinking about an adventure with weather and baking bread today",
                                              session_id="s9", exclude_ids=["c1"])
            return await worker._add_lived_memories(bundle, lived_task=task, remaining_s=5.0, corr_id="c1")

        debug, count, timed_out = asyncio.run(go())
    finally:
        worker.set_lived_memory(None)
    assert (count, timed_out, debug["backend"]) == (0, False, "degraded")
    assert lm.DEGRADED_EMPTY_NOTE in bundle.rendered and bundle.items == []
    assert seen and seen[0][0].startswith("thinking about an adventure")


def test_worker_falls_back_to_exact_words_when_the_backstop_fires(no_dense):
    from app import worker
    from orion.core.contracts.recall import MemoryBundleStatsV1, MemoryBundleV1

    class Hung(LivedMemoryRecall):
        async def recall(self, *_a, **_kw):
            await asyncio.sleep(10)

    reader = Hung(Pool({"episode_memory": [GAME, OFFICE]}), _cfg(), reranker=overlap_reranker, clock=lambda: NOW)
    asyncio.run(reader.snapshot(force=True))
    worker.set_lived_memory(reader)
    msg = "Juniper reached the final dungeon of the castle adventure game today"
    try:
        bundle = MemoryBundleV1(rendered="", items=[], stats=MemoryBundleStatsV1())

        async def go():
            task = worker._start_lived_recall(msg)
            return await worker._add_lived_memories(bundle, lived_task=task, remaining_s=0.05, corr_id="c",
                                                    query_text=msg)

        debug, count, timed_out = asyncio.run(go())
    finally:
        worker.set_lived_memory(None)
    assert (count, timed_out, debug["backend"]) == (1, True, "degraded")
    assert bundle.rendered.startswith(lm.DEGRADED_HEADER) and [i.id for i in bundle.items] == ["episode_memory:g1"]


# --- chunked rerank keeps what finished (issue #3662) -----------------------------------------

FILLER = [_row(f"f{i}", f"A note about the castle on day {i} of the week.") for i in range(12)]
CASTLE_TURN = "Juniper reached the final dungeon of the castle adventure game today"


def _chunked(stall_after: int, *, stall: str = "sleep", calls: List[int] | None = None):
    """Overlap reranker that answers ``stall_after`` calls, then sleeps past any deadline or raises."""
    seen = calls if calls is not None else []

    def rr(query, docs):
        async def go():
            seen.append(len(docs))
            if len(seen) > stall_after:
                if stall == "raise":
                    raise ConnectionError("reranker went away")
                await asyncio.sleep(10)
            return await overlap_reranker(query, docs)
        return go()

    return rr


def test_rerank_goes_in_chunks_best_retrieved_first(no_dense):
    calls: List[int] = []
    reader = LivedMemoryRecall(Pool({"episode_memory": [GAME] + FILLER}), _cfg(link_top_k=0),
                               reranker=_chunked(99, calls=calls), clock=lambda: NOW)
    out = asyncio.run(reader.recall(CASTLE_TURN))
    n = len(out.debug["candidate_ids"])
    assert n > lm.RERANK_CHUNK and calls == [lm.RERANK_CHUNK, n - lm.RERANK_CHUNK]
    assert out.backend == "hybrid" and "rerank_partial" not in out.debug
    assert out.chosen[0].rec.record_id == "g1"


def test_rerank_deadline_keeps_the_chunks_already_scored(no_dense):
    reader = LivedMemoryRecall(Pool({"episode_memory": [GAME] + FILLER}),
                               _cfg(timeout_sec=0.8, link_top_k=0), reranker=_chunked(1), clock=lambda: NOW)
    out = asyncio.run(reader.recall(CASTLE_TURN))
    # Before #3662 the whole pool was lost and this turn fell back to exact words.
    assert out.backend == "hybrid" and "degraded" not in out.debug
    n = len(out.debug["candidate_ids"])
    assert out.debug["rerank_partial"] == {"scored": lm.RERANK_CHUNK, "of": n, "reason": "timeout"}
    assert len(out.reranked) == lm.RERANK_CHUNK and out.chosen[0].rec.record_id == "g1"
    assert not out.chosen[0].degraded and out.rendered.startswith(lm.HEADER)
    assert lm.trace_of(out.debug)["rerank_partial"]["scored"] == lm.RERANK_CHUNK
    assert lm.status_of(out.debug, len(out.chosen)) == lm.STATUS_OK
    assert out.debug["timings_ms"]["total"] < 1100


def test_rerank_error_after_a_chunk_keeps_it(no_dense):
    reader = LivedMemoryRecall(Pool({"episode_memory": [GAME] + FILLER}), _cfg(link_top_k=0),
                               reranker=_chunked(1, stall="raise"), clock=lambda: NOW)
    out = asyncio.run(reader.recall(CASTLE_TURN))
    assert out.backend == "hybrid" and out.debug["rerank_partial"]["reason"] == "error"
    assert "rerank_error" in out.debug and out.chosen[0].rec.record_id == "g1"


def test_rerank_that_scores_no_chunk_still_degrades(no_dense):
    reader = LivedMemoryRecall(Pool({"episode_memory": [GAME] + FILLER}),
                               _cfg(timeout_sec=0.6, link_top_k=0), reranker=_chunked(0), clock=lambda: NOW)
    out = asyncio.run(reader.recall(CASTLE_TURN))
    assert out.debug["degraded"] == ["rerank_timeout"] and "rerank_partial" not in out.debug
    assert out.backend == "degraded" and out.chosen and out.chosen[0].degraded


TWO_WORD_FILLER = [_row(f"t{i}", f"Another castle adventure from week {i}.") for i in range(12)]


def _first_chunk_fits_nothing_then_stalls(query, docs, _seen=[]):  # noqa: B006 -- per-test reset below
    async def go():
        _seen.append(len(docs))
        if len(_seen) > 1:
            await asyncio.sleep(10)
        return [-10.0] * len(docs)
    return go()


def test_partial_rerank_with_nothing_above_the_bar_degrades_and_says_so(no_dense):
    """#3662/#3663: a part-scored pool that served nothing is not a full search: never a silent below-bar."""
    seen: List[int] = []

    def rr(query, docs):
        return _first_chunk_fits_nothing_then_stalls(query, docs, seen)

    reader = LivedMemoryRecall(Pool({"episode_memory": [GAME] + TWO_WORD_FILLER}),
                               _cfg(timeout_sec=0.8, link_top_k=0), reranker=rr, clock=lambda: NOW)
    out = asyncio.run(reader.recall(CASTLE_TURN))
    assert out.backend == "degraded" and out.debug["degraded"] == ["rerank_partial"]
    scored = {r.doc_id for r, _l in out.reranked}
    assert out.chosen and all(s.degraded and s.rec.doc_id not in scored for s in out.chosen)
    assert out.rendered.startswith(lm.DEGRADED_HEADER)
    assert lm.status_of(out.debug, len(out.chosen)) == lm.STATUS_DEGRADED


def test_partial_rerank_that_finds_nothing_at_all_leaves_the_not_proof_note(no_dense):
    seen: List[int] = []

    def rr(query, docs):
        return _first_chunk_fits_nothing_then_stalls(query, docs, seen)

    reader = LivedMemoryRecall(Pool({"episode_memory": [GAME] + FILLER}),
                               _cfg(timeout_sec=0.8, link_top_k=0), reranker=rr, clock=lambda: NOW)
    out = asyncio.run(reader.recall(CASTLE_TURN))
    assert out.debug["degraded"] == ["rerank_partial"] and not out.chosen
    assert out.rendered == lm.DEGRADED_EMPTY_NOTE
    assert lm.status_of(out.debug, 0) == lm.STATUS_DEGRADED


def test_know_shows_what_a_partial_rerank_left_unscored_as_weak(no_dense):
    reader = LivedMemoryRecall(Pool({"episode_memory": [GAME] + FILLER}), _cfg(link_top_k=0),
                               reranker=_chunked(1), clock=lambda: NOW)
    got = asyncio.run(reader.know("castle adventure game dungeon", max_hits=20, timeout_sec=0.8))
    assert [s.rec.record_id for s in got.strong] == ["g1"]
    scored = {r.doc_id for r, _l in got.recall.reranked}
    unscored_weak = [s for s in got.weak if s.rec.doc_id not in scored]
    assert unscored_weak and all(s.degraded for s in unscored_weak)


def test_exact_word_only_hit_is_in_the_first_chunk(monkeypatch):
    """Meaning hits and exact-word hits take turns in the rerank order, so a name-only hit is not
    left for the last chunk past a short deadline."""
    near = [_row(f"n{i}", f"Something about the weather number {i}.") for i in range(12)]
    biscuit = _row("b1", "Biscuit the cat came home from the vet with stitches.")

    async def embed(*_a, **_k):
        return [0.0], "m"

    async def nearest(*_a, **_k):
        return [(f"episode_memory:n{i}", 0.9 - i * 0.01) for i in range(12)]

    monkeypatch.setattr(lm, "embed", embed)
    monkeypatch.setattr(lm, "nearest", nearest)
    reader = LivedMemoryRecall(Pool({"episode_memory": near + [biscuit]}), _cfg(timeout_sec=0.8, link_top_k=0),
                               reranker=_chunked(1), clock=lambda: NOW)
    out = asyncio.run(reader.recall("did Biscuit come home with stitches from the vet visit"))
    assert out.debug["rerank_partial"]["scored"] == lm.RERANK_CHUNK
    assert "episode_memory:b1" in {r.doc_id for r, _l in out.reranked}
    assert [s.rec.record_id for s in out.chosen][:1] == ["b1"]


def test_http_rerank_goes_in_chunks_each_with_a_live_timeout(no_dense):
    import httpx

    seen: List[tuple[int, float]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        import json as _json

        docs = _json.loads(request.content)["documents"]
        seen.append((len(docs), float(request.extensions["timeout"]["read"])))
        return httpx.Response(200, json={"scores": [0.0] * len(docs)})

    rows = [_row(f"h{i}", f"Castle adventure game note {i}.") for i in range(20)]
    reader = LivedMemoryRecall(Pool({"episode_memory": rows}), _cfg(link_top_k=0), clock=lambda: NOW)

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await reader.recall(CASTLE_TURN, client=client)

    out = asyncio.run(go())
    n = len(out.debug["candidate_ids"])
    sizes = [lm.RERANK_CHUNK] * (n // lm.RERANK_CHUNK) + ([n % lm.RERANK_CHUNK] if n % lm.RERANK_CHUNK else [])
    assert [s for s, _t in seen] == sizes and len(seen) > 1
    assert all(0 < t <= 2.0 for _s, t in seen)
    assert out.backend == "hybrid" and len(out.reranked) == n
