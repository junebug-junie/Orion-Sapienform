"""``POST /lived/know`` (issue #3535) and the persisted lived-recall status (issue #3628).

Made-up records only; no DB, bus or model. Reuses the step 2 fakes.
"""
from __future__ import annotations

import asyncio
from typing import Any, List

import pytest

import app.lived_memory as lm
from app.lived_memory import LivedMemoryRecall, status_of, trace_of
from test_lived_memory_step2 import GAME, NOW, OFFICE, Pool, _cfg, _row, fixed_rewriter, overlap_reranker

from test_lived_memory_step2 import no_dense  # noqa: F401 -- fixture

STITCHES = _row("p1", "Juniper's dog came home from the vet with stitches after a small surgery.")
GARDEN = _row("p2", "Juniper planted tomatoes along the garden fence.")


# --- know: the lookup ------------------------------------------------------------------------

def test_know_finds_a_memory_from_two_words_the_chat_floor_would_skip(no_dense):
    calls: List[Any] = []
    reader = LivedMemoryRecall(Pool({"episode_memory": [STITCHES, GARDEN, OFFICE]}), _cfg(),
                               reranker=overlap_reranker, rewriter=fixed_rewriter(["x y"], calls), clock=lambda: NOW)
    # A chat turn this short is not searched at all (min_query_tokens=5)...
    assert asyncio.run(reader.recall("dog stitches")).debug["skipped"] == "few_content_words"
    # ...a deliberate lookup is, with no rewrite (its words already are the search).
    got = asyncio.run(reader.know("dog stitches vet"))
    assert [s.rec.record_id for s in got.strong] == ["p1"]
    assert got.status == "ok" and calls == []
    assert got.recall.debug["rewrite"]["outcome"] == "off"


def test_know_shows_below_bar_word_matches_as_weak_and_hides_meaning_only_neighbours(monkeypatch):
    # Dense search returns every record (nearest is never "nothing"); only the one sharing a word
    # with the search may appear as weak.
    async def embed(*_a, **_k):
        return [0.0], "m"

    async def nearest(*_a, **_k):
        return [("episode_memory:p2", 0.5), ("episode_memory:o1", 0.4)]

    monkeypatch.setattr(lm, "embed", embed)
    monkeypatch.setattr(lm, "nearest", nearest)
    reader = LivedMemoryRecall(Pool({"episode_memory": [GARDEN, OFFICE]}), _cfg(), reranker=overlap_reranker,
                               clock=lambda: NOW)
    got = asyncio.run(reader.know("garden roses"))  # one shared word: logit -1, below the 0.5 bar
    assert got.strong == []
    assert [s.rec.record_id for s in got.weak] == ["p2"]
    assert got.status == "below_bar"
    assert "episode_memory:o1" in got.recall.debug["candidate_ids"]  # reached the rerank, not shown


def test_know_caps_hits_strong_first(no_dense):
    rows = [_row(f"r{i}", f"castle game dungeon note number {i}") for i in range(12)]
    reader = LivedMemoryRecall(Pool({"episode_memory": rows}), _cfg(max_lines=4), reranker=overlap_reranker,
                               clock=lambda: NOW)
    got = asyncio.run(reader.know("castle dungeon", max_hits=6))
    # All 12 clear the bar: a chat turn injects 4, a lookup shows up to max_hits, all strong.
    assert len(got.strong) == 6 and got.weak == []
    assert len({s.rec.doc_id for s in got.strong}) == 6
    got = asyncio.run(reader.know("castle", max_hits=6))
    assert len(got.strong) + len(got.weak) <= 6


def test_know_degraded_shows_exact_word_hits_labelled(no_dense):
    async def down(query, docs):
        raise ConnectionError("rerank unreachable")

    reader = LivedMemoryRecall(Pool({"episode_memory": [STITCHES, GARDEN]}), _cfg(), reranker=down, clock=lambda: NOW)
    got = asyncio.run(reader.know("dog vet"))
    assert got.status == "degraded" and got.recall.backend == "degraded"
    ids = [s.rec.record_id for s in got.strong + got.weak]
    assert ids == ["p1"]


def test_know_never_shows_a_row_the_recheck_drops(no_dense):
    pool = Pool({"episode_memory": [GARDEN, OFFICE]})
    reader = LivedMemoryRecall(pool, _cfg(), reranker=overlap_reranker, clock=lambda: NOW)
    asyncio.run(reader.snapshot(force=True))
    pool.live["episode_memory"] = [OFFICE]  # superseded since the snapshot
    got = asyncio.run(reader.know("garden roses"))
    assert got.weak == [] and got.strong == []
    assert got.recall.debug["recheck_dropped"] == ["episode_memory:p2"]  # dropped by the gate, not missed
    assert trace_of(got.recall.debug)["recheck_dropped"] == ["episode_memory:p2"]
    assert got.status == "below_bar" and got.unverified == 0


def test_know_a_failed_recheck_is_unverified_never_below_bar(no_dense, monkeypatch):
    reader = LivedMemoryRecall(Pool({"episode_memory": [GARDEN, OFFICE]}), _cfg(), reranker=overlap_reranker,
                               clock=lambda: NOW)

    async def broken(*_a, **_k):
        raise RuntimeError("db gone")

    monkeypatch.setattr(reader, "_recheck", broken)
    got = asyncio.run(reader.know("garden roses"))
    assert got.weak == [] and got.unverified == 1 and got.status == "unverified"
    assert trace_of(got.recall.debug)["know_recheck_error"] == "RuntimeError"


# --- status: one per call (#3628) ------------------------------------------------------------

@pytest.mark.parametrize(
    "debug, served, want",
    [
        ({"backend": "hybrid", "injected": [{"id": "x"}]}, 1, "ok"),
        ({"backend": "hybrid"}, 0, "below_bar"),
        ({}, 0, "below_bar"),  # nothing matched at all: the search ran, nothing cleared the bar
        ({"skipped": "few_content_words"}, 0, "skipped_short"),
        ({"skipped": "empty_query"}, 0, "skipped_short"),
        ({"backend": "degraded", "degraded": ["rerank_timeout"]}, 0, "degraded"),
        ({"backend": "degraded", "degraded": ["rerank_unavailable"]}, 2, "degraded"),
        ({"skipped": "timeout"}, 0, "timeout"),
        ({"outer": "timeout", "backend": "degraded"}, 0, "timeout"),
        ({"outer": "timeout", "backend": "degraded"}, 1, "degraded"),
        ({"skipped": "error:RuntimeError"}, 0, "error"),
        ({"outer": "error:RuntimeError", "backend": "degraded"}, 0, "error"),
        ({"error": "HTTPError: boom"}, 0, "error"),
        ({"skipped": "disabled"}, 0, "off"),
    ],
)
def test_status_of(debug, served, want):
    assert status_of(debug, served) == want
    assert want in lm.STATUSES


def test_trace_carries_ids_and_queries_never_memory_text():
    debug = {"backend": "hybrid", "rewrite": {"outcome": "ok", "queries": ["castle game"]},
             "candidate_ids": ["episode_memory:a", "stance:b"], "injected": [{"id": "episode_memory:a", "fit": 0.9}],
             "reranked": [{"id": "episode_memory:a"}]}
    t = trace_of(debug)
    assert t == {"backend": "hybrid", "queries": ["castle game"], "rewrite": "ok",
                 "candidate_ids": ["episode_memory:a", "stance:b"], "injected_ids": ["episode_memory:a"]}


# --- worker: status reaches the persisted row -------------------------------------------------

def _persisted_row(monkeypatch, decision) -> dict:
    from app import worker
    from test_recall_telemetry_persist import _install

    log = _install(monkeypatch)
    worker._persist_decision(decision)
    sql, params, _t = [e for e in log if "INSERT INTO recall_telemetry" in e[0]][0]
    cols = [c.strip() for c in sql.split("(", 1)[1].split(")", 1)[0].split(",")]
    return dict(zip(cols, params))


@pytest.mark.parametrize("status", ["ok", "below_bar", "skipped_short", "degraded", "timeout", "error"])
def test_each_status_lands_in_the_persisted_row(monkeypatch, status):
    from orion.core.contracts.recall import RecallDecisionV1

    lived = {"status": status, "backend": "hybrid", "candidate_ids": ["episode_memory:a"],
             "rewrite": {"outcome": "ok", "queries": ["q"]}, "injected": []}
    d = RecallDecisionV1(corr_id="c1", query="q", backend_counts={"lived_memory": 0},
                         recall_debug={"lived_memory": lived})
    row = _persisted_row(monkeypatch, d)
    assert row["lived_status"] == status
    assert row["lived_trace"].adapted["candidate_ids"] == ["episode_memory:a"]


def test_a_call_where_lived_recall_did_not_run_writes_null(monkeypatch):
    from orion.core.contracts.recall import RecallDecisionV1

    row = _persisted_row(monkeypatch, RecallDecisionV1(corr_id="c1", query="q"))
    assert row["lived_status"] is None and row["lived_trace"] is None


def test_worker_sets_status_on_a_real_lived_call(no_dense):
    """End to end through _start_lived_recall/_add_lived_memories and the classifier."""
    from app import worker
    from orion.core.contracts.recall import MemoryBundleStatsV1, MemoryBundleV1

    reader = LivedMemoryRecall(Pool({"episode_memory": [GAME, OFFICE]}), _cfg(), reranker=overlap_reranker,
                               clock=lambda: NOW)
    worker.set_lived_memory(reader)
    try:
        def run(msg):
            bundle = MemoryBundleV1(rendered="", items=[], stats=MemoryBundleStatsV1())

            async def go():
                task = worker._start_lived_recall(msg)
                return await worker._add_lived_memories(bundle, lived_task=task, remaining_s=5.0, corr_id="c",
                                                        query_text=msg)

            debug, count, _ = asyncio.run(go())
            return status_of(debug, count), debug

        hit, d1 = run("Juniper reached the final dungeon of the castle adventure game today")
        short, _ = run("hi there")
        miss, _ = run("weather forecast says rain and snow across mountains tomorrow")
    finally:
        worker.set_lived_memory(None)
    assert (hit, short, miss) == ("ok", "skipped_short", "below_bar")
    assert "episode_memory:g1" in d1["candidate_ids"]


def test_worker_backstop_timeout_is_marked_outer(no_dense):
    from app import worker
    from orion.core.contracts.recall import MemoryBundleStatsV1, MemoryBundleV1

    class Hung(LivedMemoryRecall):
        async def recall(self, *_a, **_kw):
            await asyncio.sleep(10)

    reader = Hung(Pool({"episode_memory": [OFFICE]}), _cfg(), reranker=overlap_reranker, clock=lambda: NOW)
    asyncio.run(reader.snapshot(force=True))
    worker.set_lived_memory(reader)
    msg = "a long message about mountains rivers forests and weather"
    try:
        bundle = MemoryBundleV1(rendered="", items=[], stats=MemoryBundleStatsV1())

        async def go():
            task = worker._start_lived_recall(msg)
            return await worker._add_lived_memories(bundle, lived_task=task, remaining_s=0.05, corr_id="c",
                                                    query_text=msg)

        debug, count, timed_out = asyncio.run(go())
    finally:
        worker.set_lived_memory(None)
    assert timed_out and debug["outer"] == "timeout" and status_of(debug, count) == "timeout"


# --- the HTTP endpoint ---------------------------------------------------------------------

def test_know_endpoint_returns_hits_and_logs_the_turn(monkeypatch, no_dense):
    from fastapi.testclient import TestClient

    from app import main, worker

    reader = LivedMemoryRecall(Pool({"episode_memory": [STITCHES, GARDEN]}), _cfg(), reranker=overlap_reranker,
                               clock=lambda: NOW)
    persisted: List[Any] = []

    async def persist(decision):
        persisted.append(decision)

    monkeypatch.setattr(main, "persist_decision_async", persist)
    worker.set_lived_memory(reader)
    try:
        client = TestClient(main.app)  # no `with`: lifespan (bus, pools) does not start
        resp = client.post("/lived/know", json={"words": "dog stitches vet", "turn_id": "turn-123"})
    finally:
        worker.set_lived_memory(None)
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok" and body["corr_id"] == "turn-123"
    assert body["hits"][0]["id"] == "episode_memory:p1" and body["hits"][0]["strength"] == "strong"
    assert {"kind", "purpose", "date", "text", "fit"} <= set(body["hits"][0])
    (d,) = persisted
    assert (d.corr_id, d.verb) == ("turn-123", "dig.know")
    strong = [h["id"] for h in body["hits"] if h["strength"] == "strong"]
    weak = [h["id"] for h in body["hits"] if h["strength"] == "weak"]
    assert d.selected_ids == strong  # read demand counts what cleared the bar
    assert d.profile == "lived_memory.know"
    lived = d.recall_debug["lived_memory"]
    assert lived["status"] == "ok" and trace_of(lived)["injected_ids"] == strong
    assert trace_of(lived).get("weak_ids", []) == weak


def test_know_endpoint_says_off_when_the_reader_is_not_running():
    from fastapi.testclient import TestClient

    from app import main, worker

    worker.set_lived_memory(None)
    resp = TestClient(main.app).post("/lived/know", json={"words": "dog stitches"})
    assert resp.status_code == 200 and resp.json()["status"] == "off" and resp.json()["hits"] == []
    assert resp.json()["corr_id"].startswith("adhoc:")


def test_know_endpoint_answers_busy_while_another_lookup_runs(monkeypatch):
    from fastapi.testclient import TestClient

    from app import main, worker

    class Reader:
        async def know(self, *_a, **_k):
            raise AssertionError("must not run while the gate is held")

    class Held:
        def locked(self):
            return True

    monkeypatch.setattr(main, "_know_gate", Held())
    worker.set_lived_memory(Reader())
    try:
        resp = TestClient(main.app).post("/lived/know", json={"words": "dog stitches", "turn_id": "t"})
    finally:
        worker.set_lived_memory(None)
    assert resp.json()["status"] == "busy" and resp.json()["hits"] == []


def test_know_endpoint_failure_is_a_status_not_an_empty_answer(monkeypatch):
    from fastapi.testclient import TestClient

    from app import main, worker

    class Broken:
        async def know(self, *_a, **_k):
            raise RuntimeError("chroma gone")

    async def persist(decision):
        pass

    monkeypatch.setattr(main, "persist_decision_async", persist)
    worker.set_lived_memory(Broken())
    try:
        resp = TestClient(main.app).post("/lived/know", json={"words": "dog stitches"})
    finally:
        worker.set_lived_memory(None)
    assert resp.status_code == 200 and resp.json()["status"] == "error" and resp.json()["hits"] == []
