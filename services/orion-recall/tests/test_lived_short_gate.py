"""Short-message gate for lived-memory recall (issue #3613).

A message under the content-word floor searches only when it names a known entity, or when the
short-message rewrite says a reply needs a memory. Pure small talk never searches. Made-up records
and names only; no DB, bus or model (fakes only).
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

import pytest

import app.lived_memory as lm
from app.lived_entities import EntityIndex, load_entities
from app.lived_memory import (
    GATE_SHORT_ENTITY,
    GATE_SHORT_NO_CONTENT,
    GATE_SHORT_NONE,
    GATE_SHORT_REWRITE,
    GATE_WORDS,
    LivedMemoryConfig,
    LivedMemoryRecall,
    status_of,
    tokenize,
    trace_of,
)
from app.lived_rewrite import PROMPT, SHORT_PROMPT, ContextTurn, build_prompt
from orion.introspect.semantic_index import SearchConfig

NOW = datetime(2026, 11, 16, 12, 0, tzinfo=timezone.utc)


def _row(rid: str, text: str) -> Dict[str, Any]:
    at = NOW - timedelta(days=2)
    return {"record_id": rid, "purpose": "happened", "text": text, "occurred_at": at, "importance": 0.8,
            "is_correction": False, "confirmed": False, "label": None, "created_at": at}


# Made-up world: a host "vesper" with its own memories, a pet "Pickle Moss", a participant pair.
VESPER = _row("v1", "The vesper box ran hot all night and Juniper moved it to the closet shelf.")
PROMO = _row("p1", "Juniper was promoted to team lead at work in October.")
OFFICE = _row("o1", "The office was loud all evening.")

# The participant "orion" sits on only 12 memories (not a hub), so only its identity can block it.
DEGREES = [("project:vesper", 3), ("person:pickle-moss", 2), ("person:juniper", 90), ("person:orion", 12),
           ("concept:attention-work", 4), ("pr:1234", 1), ("person:juniper-sister", 1), ("project:curiosity", 5)]
ALIASES = [("vesper", "project:vesper"), ("the vesper box", "project:vesper"), ("pickle moss", "person:pickle-moss"),
           ("juniper", "person:juniper"), ("orion", "person:orion"), ("or on", "person:orion"),
           ("attention work", "concept:attention-work"), ("that", "project:vesper"), ("1234", "pr:1234"),
           ("juniper sister", "person:juniper-sister")]
PERSONS = [("Juniper", "interlocutor"), ("Orion", "self"), ("Rowan Vale", "other"), ("Mail carrier", "other")]


class Pool:
    """asyncpg-shaped fake: serving queries plus the entity tables."""

    def __init__(self, live: Dict[str, List[Dict[str, Any]]], *, degrees=DEGREES, aliases=ALIASES,
                 persons=PERSONS) -> None:
        self.live, self.degrees, self.aliases, self.persons = live, degrees, aliases, persons

    async def fetch(self, sql: str, *args: Any) -> List[Dict[str, Any]]:
        if "count(*)" in sql and "episode_memory_referent" in sql:
            return [{"referent_key": k, "n": n} for k, n in self.degrees]
        if "FROM episode_memory_referent" in sql:
            return []
        if "FROM referent_alias" in sql:
            return [{"alias": a, "referent_key": k} for a, k in self.aliases]
        if "FROM person\n" in sql:
            return [{"display_name": d, "role": r} for d, r in self.persons]
        if "FROM chat_history_log" in sql:
            return []
        kind = ("episode_memory" if "FROM episode_memory" in sql else
                "stance" if "FROM memory_crystallizations" in sql else "person_fact")
        rows = list(self.live.get(kind, []))
        if args:
            rows = [r for r in rows if r["record_id"] in set(args[0])]
        return rows


def _cfg(**kw: Any) -> LivedMemoryConfig:
    base = dict(search=SearchConfig("http://chroma", "http://embed", "orion_lived_memory", 0.0),
                rerank_url="http://rerank", min_fit=0.5, max_lines=4, timeout_sec=2.0, min_query_tokens=5,
                rewrite_timeout_sec=0.5, link_top_k=0)
    base.update(kw)
    return LivedMemoryConfig(**base)


def overlap_reranker(query: str, docs: List[str]):
    q = set(tokenize(query))

    async def go():
        return [(-4.0 + 3.0 * len(q & set(tokenize(d)))) for d in docs]

    return go()


class Rewriter:
    def __init__(self, short_answer: List[str], general_answer: List[str] | None = None) -> None:
        self.short_answer, self.general_answer = short_answer, general_answer or []
        self.calls: List[str] = []

    async def __call__(self, message, context, short=False):
        self.calls.append("short" if short else "general")
        return list(self.short_answer if short else self.general_answer)


@pytest.fixture
def dense_calls(monkeypatch):
    calls: List[str] = []

    async def embed(_client, _cfg, text, **_k):
        calls.append(text)
        return [0.0], "m"

    async def nearest(*_a, **_k):
        return []

    monkeypatch.setattr(lm, "embed", embed)
    monkeypatch.setattr(lm, "nearest", nearest)
    return calls


def _reader(rewriter=None, **kw) -> LivedMemoryRecall:
    return LivedMemoryRecall(Pool({"episode_memory": [VESPER, PROMO, OFFICE]}), _cfg(**kw),
                             reranker=overlap_reranker, rewriter=rewriter, clock=lambda: NOW)


# --- the entity index -------------------------------------------------------------------------

def test_entity_index_names_entities_from_the_stores_and_drops_junk():
    ix = EntityIndex(DEGREES, ALIASES, PERSONS)
    assert ix.mentions("what's up with vesper?") == {"project:vesper"}
    assert ix.mentions("how is Pickle Moss doing") == {"person:pickle-moss"}
    assert ix.mentions("did Pickle eat?") == {"person:pickle-moss"}           # first name of a person referent
    assert ix.mentions("any news from Rowan?") == {"person_name:rowan vale"}  # person table, first name
    # A role-like display name counts whole, never by its first word.
    assert ix.mentions("check the mail") == set()
    assert ix.mentions("the mail carrier came") == {"person_name:mail carrier"}
    # A key slug with no name alias is not a name ("project:curiosity").
    assert ix.mentions("just out of curiosity") == set()
    # Participants never count: not by name, not by another alias of theirs, not as another
    # entity's first word ("juniper sister" still counts whole).
    assert ix.mentions("thanks Orion") == set()
    assert ix.mentions("turn it or on") == set()
    assert ix.mentions("juniper here") == set()
    assert ix.mentions("my juniper sister called") == {"person:juniper-sister"}
    # Topics, PR numbers and function-word aliases are not entities.
    assert ix.mentions("attention work") == set()
    assert ix.mentions("about 1234 of them") == set()
    assert ix.mentions("that one") == set()
    # Whole words only.
    assert ix.mentions("vespers and evening") == set()


def test_without_participant_rows_the_degree_backstop_blocks_hubs():
    ix = EntityIndex([("person:juniper", 90), ("project:vesper", 3)], [("juniper", "person:juniper"),
                                                                        ("vesper", "project:vesper")], [])
    assert ix.mentions("juniper and vesper") == {"project:vesper"}


def test_entity_index_loads_from_the_pool():
    ix = asyncio.run(load_entities(Pool({})))
    assert "vesper" in ix.names and "juniper" not in ix.names and "orion" not in ix.names


# --- the gate ---------------------------------------------------------------------------------

def test_no_content_words_never_rewrites_or_searches(dense_calls):
    rw = Rewriter(["anything at all here"])
    reader = _reader(rw)
    for msg in ("ok", "lol", "thanks!", "haha yeah"):
        out = asyncio.run(reader.recall(msg))
        assert out.lines == [] and out.debug["gate"] == GATE_SHORT_NO_CONTENT
        assert status_of(out.debug, 0) == "skipped_short"
    assert rw.calls == [] and dense_calls == []


def test_a_short_message_naming_an_entity_searches(dense_calls):
    rw = Rewriter([], ["vesper box closet heat"])
    out = asyncio.run(_reader(rw).recall("what's up with vesper?"))
    assert out.debug["gate"] == GATE_SHORT_ENTITY
    assert [s.rec.record_id for s in out.chosen] == ["v1"]
    assert rw.calls == ["general"]          # the usual rewrite, not the short question
    assert dense_calls                      # a real search ran
    assert status_of(out.debug, len(out.chosen)) == "ok"
    assert trace_of(out.debug)["gate"] == GATE_SHORT_ENTITY


def test_a_short_entity_message_searches_without_a_rewriter(dense_calls):
    out = asyncio.run(_reader(None).recall("vesper closet again?"))
    assert out.debug["gate"] == GATE_SHORT_ENTITY and [s.rec.record_id for s in out.chosen] == ["v1"]


def test_a_short_question_the_rewrite_says_needs_memory_searches(dense_calls):
    rw = Rewriter(["Juniper promoted to team lead"])
    out = asyncio.run(_reader(rw).recall("When did I get promoted?"))
    assert out.debug["gate"] == GATE_SHORT_REWRITE and rw.calls == ["short"]
    assert [s.rec.record_id for s in out.chosen] == ["p1"]
    assert trace_of(out.debug)["queries"] == ["Juniper promoted to team lead"]


def test_short_small_talk_the_rewrite_calls_none_never_searches(dense_calls):
    rw = Rewriter([])
    out = asyncio.run(_reader(rw).recall("sounds lovely, enjoy!"))
    assert out.lines == [] and out.debug["gate"] == GATE_SHORT_NONE and out.debug["skipped"] == "few_content_words"
    assert rw.calls == ["short"] and dense_calls == []    # not even the embedder was called
    assert status_of(out.debug, 0) == "skipped_short"
    assert trace_of(out.debug)["gate"] == GATE_SHORT_NONE


def test_a_failing_or_slow_short_rewrite_means_no_search(dense_calls):
    async def broken(message, context, short=False):
        raise RuntimeError("gateway refused")

    async def slow(message, context, short=False):
        await asyncio.sleep(5)
        return ["never used"]

    for rw in (broken, slow):
        out = asyncio.run(_reader(rw, rewrite_timeout_sec=0.05).recall("When did I get promoted?"))
        assert out.debug["gate"] == GATE_SHORT_NONE and out.lines == []
    assert dense_calls == []


def test_without_a_rewriter_a_short_message_with_no_entity_is_skipped(dense_calls):
    out = asyncio.run(_reader(None).recall("When did I get promoted?"))
    assert out.debug["gate"] == GATE_SHORT_NONE and dense_calls == []


def test_long_messages_are_unchanged(dense_calls):
    rw = Rewriter(["should not be asked"], [])
    out = asyncio.run(_reader(rw).recall("Juniper was promoted to team lead at work, remember that day?"))
    assert out.debug["gate"] == GATE_WORDS and rw.calls == ["general"]
    assert [s.rec.record_id for s in out.chosen] == ["p1"]


def test_degraded_now_serves_a_short_entity_message():
    reader = _reader(None)
    asyncio.run(reader.snapshot(force=True))
    out = asyncio.run(reader.degraded_now("vesper again?", reason="timeout"))   # one word + a name
    assert [s.rec.record_id for s in out.chosen] == ["v1"]
    out = asyncio.run(reader.degraded_now("thanks a lot", reason="timeout"))
    assert out.chosen == [] and out.backend == ""


def test_the_switch_off_is_the_plain_floor(dense_calls):
    rw = Rewriter(["Juniper promoted to team lead"], ["vesper"])
    reader = _reader(rw, short_gate=False)
    for msg in ("what's up with vesper?", "When did I get promoted?"):
        out = asyncio.run(reader.recall(msg))
        assert out.debug["gate"] == "short_off" and out.lines == [] and status_of(out.debug, 0) == "skipped_short"
    assert rw.calls == [] and dense_calls == []
    # The degraded fallback follows the same switch.
    asyncio.run(reader.snapshot(force=True))
    out = asyncio.run(reader.degraded_now("vesper again?", reason="timeout"))
    assert out.chosen == [] and out.debug["gate"] == "short_off"


def test_a_failed_entity_reload_keeps_the_last_good_index(dense_calls):
    class Flaky(Pool):
        fail = False

        async def fetch(self, sql, *args):
            if self.fail and "referent_alias" in sql:
                raise RuntimeError("db hiccup")
            return await super().fetch(sql, *args)

    pool = Flaky({"episode_memory": [VESPER]})
    reader = LivedMemoryRecall(pool, _cfg(), reranker=overlap_reranker, rewriter=Rewriter([]), clock=lambda: NOW)
    asyncio.run(reader.snapshot(force=True))
    pool.fail = True
    asyncio.run(reader.snapshot(force=True))
    out = asyncio.run(reader.recall("vesper closet again?"))
    assert out.debug["gate"] == GATE_SHORT_ENTITY and [s.rec.record_id for s in out.chosen] == ["v1"]


def test_the_degraded_fallback_records_its_gate():
    reader = _reader(None)
    asyncio.run(reader.snapshot(force=True))
    assert trace_of(asyncio.run(reader.degraded_now("vesper again?", reason="timeout")).debug)["gate"] == GATE_SHORT_ENTITY
    assert trace_of(asyncio.run(reader.degraded_now("thanks a lot", reason="timeout")).debug)["gate"] == GATE_SHORT_NONE
    assert trace_of(asyncio.run(reader.degraded_now("ok", reason="timeout")).debug)["gate"] == GATE_SHORT_NO_CONTENT
    long = "the vesper box ran hot all night on the closet shelf"
    assert trace_of(asyncio.run(reader.degraded_now(long, reason="timeout")).debug)["gate"] == GATE_WORDS


def test_the_trace_counts_entities_the_short_searches_name(dense_calls):
    out = asyncio.run(_reader(Rewriter(["vesper box closet heat"])).recall("is it still too hot?"))
    t = trace_of(out.debug)
    assert t["gate"] == GATE_SHORT_REWRITE and t["rewrite_entities"] == 1


def test_an_entity_index_failure_falls_to_the_rewrite(dense_calls):
    class Broken(Pool):
        async def fetch(self, sql, *args):
            if "referent_alias" in sql:
                raise RuntimeError("db down")
            return await super().fetch(sql, *args)

    rw = Rewriter([])
    reader = LivedMemoryRecall(Broken({"episode_memory": [VESPER]}), _cfg(), reranker=overlap_reranker,
                               rewriter=rw, clock=lambda: NOW)
    out = asyncio.run(reader.recall("what's up with vesper?"))
    assert out.debug["gate"] == GATE_SHORT_NONE and rw.calls == ["short"]


def test_the_short_prompt_is_a_retrieval_need_question():
    p = build_prompt("how tall am I?", [ContextTurn("hi", "hello")], short=True)
    assert p.startswith(SHORT_PROMPT.split("\n", 1)[0]) and "NONE" in p and "already answers" in p
    assert build_prompt("x y", []).startswith(PROMPT.split("\n", 1)[0])
