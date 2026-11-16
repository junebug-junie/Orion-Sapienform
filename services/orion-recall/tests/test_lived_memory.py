"""Lived-memory recall (issue #3533). Made-up records only."""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

import pytest

import app.lived_memory as lm
from app.lived_memory import (
    BM25,
    LivedMemoryConfig,
    LivedMemoryRecall,
    name_tokens,
    rank,
    render_line,
    tokenize,
)
from orion.introspect.semantic_index import SearchConfig
from orion.memory import lived_memory as store
from orion.memory.lived_memory import LivedRecord

NOW = datetime(2026, 11, 16, 12, 0, tzinfo=timezone.utc)


def _rec(rid: str, text: str, *, kind: str = "episode_memory", days_ago: float = 1.0, importance: float = 0.8,
         correction: bool = False) -> LivedRecord:
    at = NOW - timedelta(days=days_ago)
    return LivedRecord(kind=kind, record_id=rid, purpose="happened" if kind == "episode_memory" else kind,
                       text=text, occurred_at=at, importance=importance, is_correction=correction, created_at=at)


class FakePool:
    """asyncpg-shaped: answers the three serving queries from an in-memory 'live' table."""

    def __init__(self, live: Dict[str, List[Dict[str, Any]]]) -> None:
        self.live = live

    async def fetch(self, sql: str, *args: Any) -> List[Dict[str, Any]]:
        kind = ("episode_memory" if "FROM episode_memory" in sql else
                "stance" if "FROM memory_crystallizations" in sql else "person_fact")
        rows = list(self.live.get(kind, []))
        if args:
            ids = set(args[0])
            rows = [r for r in rows if r["record_id"] in ids]
        return rows


def _row(rid: str, text: str, **kw: Any) -> Dict[str, Any]:
    return {"record_id": rid, "purpose": kw.get("purpose", "happened"), "text": text,
            "occurred_at": kw.get("at", NOW - timedelta(days=2)), "importance": kw.get("importance", 0.8),
            "is_correction": kw.get("correction", False), "confirmed": False, "label": kw.get("label"),
            "created_at": kw.get("at", NOW - timedelta(days=2))}


def _cfg(min_fit: float = 0.5, max_lines: int = 4) -> LivedMemoryConfig:
    return LivedMemoryConfig(search=SearchConfig("http://chroma", "http://embed", "orion_lived_memory", 0.0),
                             rerank_url="http://rerank", min_fit=min_fit, max_lines=max_lines, timeout_sec=1.0)


def _overlap_reranker(query: str, docs: List[str]):
    """Fake cross-encoder: logit grows with shared content words."""
    q = set(tokenize(query))

    async def go():
        return [(-4.0 + 3.0 * len(q & set(tokenize(d)))) for d in docs]

    return go()


@pytest.fixture
def no_dense(monkeypatch):
    async def embed(*_a, **_k):
        return [0.0], "m"

    async def nearest(*_a, **_k):
        return []

    monkeypatch.setattr(lm, "embed", embed)
    monkeypatch.setattr(lm, "nearest", nearest)


def test_tokenize_drops_stopwords_and_possessives():
    assert tokenize("Juniper's dog had his STITCHES removed") == ["juniper", "dog", "stitches", "removed"]


def test_name_tokens_are_capitalized_mid_sentence_or_known_people():
    assert name_tokens("my pup Biscuit had stitches. Then we walked") == {"biscuit"}
    assert name_tokens("talked with robin today", ["Robin Vale"]) == {"robin"}


def test_bm25_weights_names():
    bm = BM25(["the dog went to the vet", "Biscuit went to the vet"])
    plain = bm.scores(["biscuit", "vet"])
    named = bm.scores(["biscuit", "vet"], {"biscuit": 2.5})
    assert named[1] > plain[1] > plain[0]


def test_rank_applies_the_bar_then_orders_by_fit_recency_importance():
    old = _rec("a", "old fit", days_ago=200)
    new = _rec("b", "new fit", days_ago=1)
    weak = _rec("c", "weak")
    out = rank([(old, 2.0), (new, 2.0), (weak, -6.0)], min_fit=0.5, max_lines=4, now=NOW)
    assert [s.rec.record_id for s in out] == ["b", "a"]
    assert rank([(weak, -6.0)], min_fit=0.5, max_lines=4, now=NOW) == []


def test_a_correction_outranks_an_equally_fitting_memory():
    plain = _rec("p", "x", importance=0.8)
    fix = _rec("f", "y", importance=0.8, correction=True)
    assert rank([(plain, 1.0), (fix, 1.0)], min_fit=0.1, max_lines=4, now=NOW)[0].rec.record_id == "f"


def test_rank_caps_lines():
    recs = [(_rec(str(i), "t"), 3.0) for i in range(9)]
    assert len(rank(recs, min_fit=0.1, max_lines=4, now=NOW)) == 4


def test_render_line_names_kind_record_id_and_date():
    line = render_line(_rec("8b51", "Robin's dog had a small surgery.", days_ago=5))
    assert line.startswith("- (episode_memory/happened · episode_memory:8b51 · 2026-11-11) Robin")


def test_person_fact_text_names_who_it_is_about():
    rec = store.record_from_row("person_fact", _row("f1", "she/her", purpose="pronouns", label="Robin Vale"))
    assert rec.text == "Robin Vale, pronouns: she/her"


def test_serving_sql_excludes_superseded_rejected_and_expired_memories():
    sql = store.EPISODE_SQL
    assert "status = 'active'" in sql and "confirmation_state <> 'rejected'" in sql
    assert "expires_at" in sql and "c.supersedes_memory_id = m.memory_id" in sql
    assert "removed_at IS NULL" in store.PERSON_FACT_SQL
    assert "kind = 'stance' AND c.status = 'active'" in store.STANCE_SQL


def test_recall_finds_by_exact_words_reranks_and_renders(no_dense):
    pool = FakePool({"episode_memory": [
        _row("m1", "Juniper told me Biscuit had surgery and got stitches."),
        _row("m2", "The office was loud all evening."),
    ]})
    reader = LivedMemoryRecall(pool, _cfg(min_fit=0.5), reranker=_overlap_reranker, clock=lambda: NOW)
    out = asyncio.run(reader.recall("Biscuit got his stitches out today!"))
    assert [s.rec.record_id for s in out.chosen] == ["m1"]
    assert out.rendered.startswith(lm.HEADER) and "episode_memory:m1" in out.rendered
    assert lm.items_for(out.chosen)[0]["id"] == "episode_memory:m1"


def test_small_talk_injects_nothing(no_dense):
    pool = FakePool({"episode_memory": [_row("m1", "Juniper told me Biscuit had surgery and got stitches.")]})
    reader = LivedMemoryRecall(pool, _cfg(min_fit=0.5), reranker=_overlap_reranker, clock=lambda: NOW)
    out = asyncio.run(reader.recall("hey, how are you?"))
    assert out.lines == [] and out.rendered == ""


def test_a_dense_hit_for_a_record_no_longer_live_is_never_served(monkeypatch):
    async def embed(*_a, **_k):
        return [0.0], "m"

    async def nearest(*_a, **_k):
        return [("episode_memory:gone", 0.99), ("episode_memory:m1", 0.8)]

    monkeypatch.setattr(lm, "embed", embed)
    monkeypatch.setattr(lm, "nearest", nearest)
    pool = FakePool({"episode_memory": [_row("m1", "Biscuit had stitches.")]})
    reader = LivedMemoryRecall(pool, _cfg(min_fit=0.0), reranker=_overlap_reranker, clock=lambda: NOW)
    out = asyncio.run(reader.recall("Biscuit stitches"))
    assert [s.rec.doc_id for s in out.chosen] == ["episode_memory:m1"]
    assert out.debug["candidates"]["dense_not_live"] == 1


def test_a_record_superseded_after_the_snapshot_is_dropped_at_recheck(no_dense):
    pool = FakePool({"episode_memory": [_row("m1", "Biscuit had stitches.")]})
    reader = LivedMemoryRecall(pool, _cfg(min_fit=0.0), reranker=_overlap_reranker, clock=lambda: NOW)
    asyncio.run(reader.snapshot(force=True))
    pool.live["episode_memory"] = []  # a correction landed: the serving query no longer returns it
    out = asyncio.run(reader.recall("Biscuit stitches"))
    assert out.lines == [] and out.debug["recheck_dropped"] == ["episode_memory:m1"]


def test_as_of_hides_records_stored_after_the_replayed_message(no_dense):
    later = _row("m2", "Biscuit had stitches removed.", at=NOW + timedelta(hours=1))
    pool = FakePool({"episode_memory": [_row("m1", "Biscuit had surgery and stitches."), later]})
    reader = LivedMemoryRecall(pool, _cfg(min_fit=0.0), reranker=_overlap_reranker, clock=lambda: NOW)
    out = asyncio.run(reader.recall("Biscuit stitches removed", as_of=NOW))
    assert {r.record_id for r, _ in out.reranked} == {"m1"}


def test_embedder_down_still_serves_exact_word_matches(monkeypatch):
    from orion.introspect.semantic_index import SearchUnavailableError

    async def embed(*_a, **_k):
        raise SearchUnavailableError("embedder unavailable: ConnectError")

    monkeypatch.setattr(lm, "embed", embed)
    pool = FakePool({"episode_memory": [_row("m1", "Biscuit had surgery and stitches.")]})
    reader = LivedMemoryRecall(pool, _cfg(min_fit=0.5), reranker=_overlap_reranker, clock=lambda: NOW)
    out = asyncio.run(reader.recall("Biscuit stitches"))
    assert out.lines and "dense_error" in out.debug


def test_worker_appends_lived_lines_to_the_continuity_digest(monkeypatch, no_dense):
    from app import worker
    from orion.core.contracts.recall import MemoryBundleV1, MemoryBundleStatsV1, RecallQueryV1

    pool = FakePool({"episode_memory": [_row("m1", "Juniper told me Biscuit had surgery and got stitches.")]})
    worker.set_lived_memory(LivedMemoryRecall(pool, _cfg(min_fit=0.5), reranker=_overlap_reranker, clock=lambda: NOW))
    try:
        bundle = MemoryBundleV1(rendered="[Recent thread — continuity only]\nUser: hi", items=[],
                                stats=MemoryBundleStatsV1())
        q = RecallQueryV1(fragment="Biscuit got his stitches out!", verb="chat_general")
        debug, count, hit = asyncio.run(worker._add_lived_memories(
            q, bundle, query_text=q.fragment, remaining_s=5.0, corr_id="c1"))
    finally:
        worker.set_lived_memory(None)
    assert count == 1 and not hit
    assert bundle.rendered.startswith("[Recent thread") and lm.HEADER in bundle.rendered
    assert [i.id for i in bundle.items] == ["episode_memory:m1"] and bundle.items[0].source == "lived_memory"
    assert debug["injected"][0]["id"] == "episode_memory:m1" and "text" not in str(debug["injected"])


def test_worker_skips_lived_memory_past_the_deadline():
    from app import worker
    from orion.core.contracts.recall import MemoryBundleV1, MemoryBundleStatsV1, RecallQueryV1

    class Slow:
        async def recall(self, _q):
            await asyncio.sleep(5)

    worker.set_lived_memory(Slow())
    try:
        bundle = MemoryBundleV1(rendered="", items=[], stats=MemoryBundleStatsV1())
        q = RecallQueryV1(fragment="x", verb="chat_general")
        debug, count, hit = asyncio.run(worker._add_lived_memories(q, bundle, query_text="x", remaining_s=0.05,
                                                                   corr_id="c"))
    finally:
        worker.set_lived_memory(None)
    assert (count, hit, debug["skipped"]) == (0, True, "timeout") and bundle.rendered == ""
