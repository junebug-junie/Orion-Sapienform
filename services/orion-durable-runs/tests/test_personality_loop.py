"""Personality graph loop (spec 2026-10-16 rev 2): sources, store transaction, driver tick, routes.

No database here. What is pinned: first boot reads nothing older than boot (no backfill) and seeds
from self-inquiry once; a failed embed holds its source at the previous row and the row is folded
next tick; a curiosity run re-read after its outcome is re-upserted is not weighted twice; a tick
with no evidence still persists fade; a failing tick never raises; the frame route answers 503
when the loop is disabled or its tables are missing; PERSONALITY_ENABLED=false starts nothing.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import sys
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT), str(SERVICE_ROOT)]

from orion.personality.situation import SituationInputs  # noqa: E402
from orion.personality.graph import Node  # noqa: E402

from app import personality_sources as ps  # noqa: E402
from app.personality_driver import PersonalityDriver, accept, embed_all  # noqa: E402
from app.personality_sources import Candidate, Cursor, SourceItem  # noqa: E402
from app.personality_store import (  # noqa: E402
    INSERT_EVIDENCE_SQL,
    INSERT_HISTORY_SQL,
    UPDATE_NODE_WEIGHT_SQL,
    UPSERT_CURSOR_SQL,
    UPSERT_NODE_SQL,
    PersonalityStore,
    TickWrite,
    node_key,
    split_nodes,
)

T0 = datetime(2026, 10, 16, 18, 0, tzinfo=timezone.utc)


def utc(**kw) -> datetime:
    return T0 + timedelta(**kw)


# --- sources (pure) ---------------------------------------------------------------------------


def test_readable_prior_strips_prefixes_and_date():
    assert ps.readable_prior("curiosity_prior:self:reading_link_can_cross_region_boundary_20261015") == \
        "reading link can cross region boundary"
    assert ps.readable_prior("curiosity_prior:body_heat_tracks_gpu") == "body heat tracks gpu"
    assert ps.readable_prior("self:top8_20261007") == "top8"
    assert ps.readable_prior("curiosity_prior:self:keeps_2026_date_mid") == "keeps 2026 date mid"


def test_excerpt_first_sentence_collapsed_and_capped():
    assert ps.excerpt("  I keep   returning\nto continuity. Then more text.") == "I keep returning to continuity."
    # A bold title line is dropped when body text follows (the day letter's "**Note for <date>**").
    assert ps.excerpt("**Note for 2026-10-15**\n\nToday felt like closing loops. More.") == "Today felt like closing loops."
    # Short leading fragments ("Dr.") do not end the sentence.
    assert ps.excerpt("Dr. X said the field was quiet today. Then.") == "Dr. X said the field was quiet today."
    long = "word " * 100
    out = ps.excerpt(long)
    assert len(out) <= ps.EXCERPT_MAX and out.endswith("…")
    assert ps.excerpt("") == "" and ps.excerpt(None) == ""


def test_embed_text_joins_and_caps():
    assert ps.embed_text("Title", None, "  body ") == "Title\n\nbody"
    assert len(ps.embed_text("x" * 5000)) == ps.EMBED_TEXT_MAX


def test_sql_is_keyset_capped_and_chat_is_naive_utc():
    for spec in ps.SOURCES:
        assert "%(at)s" in spec.sql and "%(ref)s" in spec.sql and "%(hi)s" in spec.sql
        assert "LIMIT %(limit)s" in spec.sql and "ORDER BY" in spec.sql
    by = {s.name: s for s in ps.SOURCES}
    for lane in ("chat", "outreach"):
        assert "AT TIME ZONE 'UTC'" in by[lane].sql
    assert "btrim(coalesce(prompt, '')) <> ''" in by["chat"].sql
    assert "btrim(coalesce(prompt, '')) = ''" in by["outreach"].sql
    assert "o.completed_at" in by["curiosity_choice"].sql and "JOIN curiosity_offer_decisions" in by["curiosity_choice"].sql
    assert "produced_by = 'curiosity_self_inquiry'" in by["self_inquiry_seed"].sql
    assert "mirror_kinds" in by["journal"].sql and set(ps.JOURNAL_MIRROR_SOURCE_KINDS) == {"world_pulse", "orion_day"}
    assert "DISTINCT ON (concept_id)" in ps.SEED_SQL
    assert set(ps.SOURCE_NAMES) == {"dream", "journal", "reverie", "reading", "day_letter", "chat", "outreach",
                                    "curiosity_choice", "self_inquiry_seed"}


def test_row_builders_ids_lanes_and_naive_chat_time():
    item = ps.chat_item("chat")({"id": "c1", "created_at": T0.replace(tzinfo=None), "response": "Hello there friend. Yes.",
                                 "correlation_id": "k", "session_id": "s"})
    (c,) = item.candidates
    assert c.evidence_id == "chat:c1" and c.lane == "chat" and c.occurred_at == T0 and c.occurred_at.tzinfo is not None
    assert item.at == T0 and item.ref == "c1"
    j = ps.journal_item({"entry_id": "e1", "created_at": T0, "title": "Metacog", "body": "The field drifted all day.",
                         "source_kind": "metacog", "mode": "x"})
    assert j.candidates[0].meta["source_kind"] == "metacog" and j.candidates[0].text.startswith("Metacog\n\n")
    assert j.candidates[0].excerpt == "The field drifted all day."
    empty = ps.reverie_item({"thought_id": "t", "created_at": T0, "interpretation": "   "})
    assert empty.candidates == () and empty.ref == "t"
    r = ps.reading_item({"seed_id": "s1", "stage2_completed_at": T0, "title": "Art", "summary": "Orion's read of it."})
    assert r.candidates[0].evidence_id == "reading:s1" and r.candidates[0].excerpt == "Orion's read of it."


def _run_row(run_id="r1", completed=T0, tested=("self:a_thing_20261015", "self:b_thing")):
    return {
        "run_id": run_id, "turn_started_at": completed - timedelta(minutes=5), "decided_at": completed - timedelta(minutes=5),
        "arm": "value", "material_ids": [],
        "offered": [{"prior_id": "self:a_thing_20261015", "rank": 1, "slot": "order", "expected_nats": 0.1},
                    {"prior_id": "self:c_other", "rank": 2, "slot": "order", "expected_nats": 0.05}],
        "stale_offered": [],
        "per_prior": [{"kind": "tested", "prior_id": p} for p in tested],
        "completed_at": completed, "turn_ok": True, "realized_nats": 0.2,
    }


def test_curiosity_item_one_evidence_per_chosen_prior():
    item = ps.curiosity_item(_run_row())
    ids = [c.evidence_id for c in item.candidates]
    assert ids == ["curiosity_choice:r1:self:a_thing_20261015", "curiosity_choice:r1:self:b_thing"]
    a, b = item.candidates
    assert a.text == "a thing" and a.lane == "curiosity_choice"
    assert a.meta == {"choice_id": "curiosity_prior_selection:r1", "chosen_best_rank": 1, "rank": 1,
                      "prior_id": "self:a_thing_20261015"}
    assert b.meta["rank"] is None
    nothing = ps.curiosity_item({**_run_row(), "per_prior": [{"kind": "formed", "prior_id": "self:new"}]})
    assert nothing.candidates == () and nothing.ref == "r1"


# --- store (fake connection) ------------------------------------------------------------------


class FakeCursor:
    def __init__(self, log):
        self.log = log

    async def execute(self, sql, params=None):
        self.log.append(("execute", sql, params))

    async def executemany(self, sql, rows):
        self.log.append(("executemany", sql, list(rows)))

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


class FakeTxn:
    def __init__(self, log):
        self.log = log

    async def __aenter__(self):
        self.log.append(("begin",))

    async def __aexit__(self, exc_type, *a):
        self.log.append(("rollback",) if exc_type else ("commit",))
        return False


class FakeConn:
    def __init__(self, log):
        self.log = log

    def transaction(self):
        return FakeTxn(self.log)

    def cursor(self):
        return FakeCursor(self.log)


class FakePool:
    def __init__(self):
        self.log = []

    def connection(self):
        pool = self

        class _CM:
            async def __aenter__(self):
                return FakeConn(pool.log)

            async def __aexit__(self, *a):
                return False

        return _CM()


def node(did="d1", *, status="candidate", weight=1.0, centroid_n=1, label="a label", updated=T0, lanes=None) -> Node:
    return Node(disposition_id=did, origin="recurrence", status=status, label=label, label_similarity=1.0,
                centroid=[1.0, 0.0], centroid_n=centroid_n, weight=weight, evidence_count=centroid_n,
                lane_counts=lanes or {"chat": 1}, chosen_count=0, first_seen_at=T0, last_reinforced_at=T0,
                retired_at=None, updated_at=updated)


def test_commit_tick_is_one_transaction_in_migration_order():
    from orion.personality.graph import EvidenceRow, HistoryRow

    pool = FakePool()
    ev = EvidenceRow("chat:c1", "chat", "chat_history_log:c1", T0, "x", "d2", 1.0, 1.0, {"a": 1})
    w = TickWrite(now=T0, evidence=[ev], nodes=[node("d1", weight=0.9), node("d2", centroid_n=1)],
                  loaded={"d1": node_key(node("d1"))}, history=[HistoryRow("d2", T0, 1.0, "candidate", 1)],
                  cursors={"chat": Cursor(T0, "c1"), "dream": Cursor(T0, "")})
    asyncio.run(PersonalityStore(pool).commit_tick(w))
    kinds = [e[0] for e in pool.log]
    assert kinds[0] == "begin" and kinds[-1] == "commit" and kinds.count("begin") == 1
    sqls = [e[1] for e in pool.log if e[0] in ("execute", "executemany")]
    assert "statement_timeout" in sqls[0]
    assert sqls[1:] == [INSERT_EVIDENCE_SQL, UPSERT_NODE_SQL, UPDATE_NODE_WEIGHT_SQL, INSERT_HISTORY_SQL, UPSERT_CURSOR_SQL,
                       UPSERT_CURSOR_SQL]  # last: the tick heartbeat the frame is dated by
    assert "ON CONFLICT (evidence_id) DO NOTHING" in INSERT_EVIDENCE_SQL
    assert "ON CONFLICT (disposition_id, at) DO NOTHING" in INSERT_HISTORY_SQL
    by_sql = {e[1]: e[2] for e in pool.log if e[0] == "executemany"}
    assert [r["disposition_id"] for r in by_sql[UPSERT_NODE_SQL]] == ["d2"]
    assert [r["disposition_id"] for r in by_sql[UPDATE_NODE_WEIGHT_SQL]] == ["d1"]
    assert json.loads(by_sql[INSERT_EVIDENCE_SQL][0]["meta"]) == {"a": 1}
    curs = {r["source"]: r for r in by_sql[UPSERT_CURSOR_SQL]}
    assert curs["chat"]["ref"] == "c1" and curs["dream"]["ref"] is None


def test_split_nodes_rewrites_centroid_only_when_folded_or_new():
    a, b = node("a", centroid_n=3, label="l"), node("b", centroid_n=3, label="l")
    loaded = {"a": node_key(a), "b": node_key(b)}
    faded_a = replace(a, weight=0.5, status="retired")
    reinforced_b = replace(b, centroid_n=4)
    full, light = split_nodes([faded_a, reinforced_b, node("c")], loaded)
    assert [r["disposition_id"] for r in full] == ["b", "c"] and [r["disposition_id"] for r in light] == ["a"]
    assert "centroid" not in light[0] and light[0]["status"] == "retired"
    assert "lane_recent" in light[0] and "lane_recent" in UPDATE_NODE_WEIGHT_SQL
    # Any column the light UPDATE does not write forces a full upsert (here lane_counts alone).
    full, light = split_nodes([replace(a, lane_counts={"chat": 1, "dream": 1})], loaded)
    assert [r["disposition_id"] for r in full] == ["a"] and light == []


# --- driver (fakes) ---------------------------------------------------------------------------


def vec_for(text: str, dim: int = 64) -> list[float]:
    raw = hashlib.sha256(text.encode()).digest() * 2
    v = [b - 127.5 for b in raw[:dim]]
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v]


class Embedder:
    def __init__(self):
        self.fail: set[str] = set()
        self.calls: list[str] = []

    async def __call__(self, doc_id: str, text: str) -> list[float]:
        self.calls.append(text)
        if text in self.fail:
            raise RuntimeError("embedder down")
        return vec_for(text)


class FakeReader:
    def __init__(self):
        self.items: dict[str, list[SourceItem]] = {}
        self.seeds: list[SourceItem] = []
        self.seed_reads = 0

    def add(self, source: str, item: SourceItem) -> None:
        self.items.setdefault(source, []).append(item)
        self.items[source].sort(key=lambda i: (i.at, i.ref))

    async def read(self, cursors, hi):
        out = {}
        for name, cur in cursors.items():
            if cur.at >= hi:
                continue
            out[name] = [i for i in self.items.get(name, []) if (i.at, i.ref) > (cur.at, cur.ref) and i.at < hi][:200]
        return out

    async def read_seeds(self, now):
        self.seed_reads += 1
        return [s for s in self.seeds if s.at <= now]


def _copy(n: Node) -> Node:
    return replace(n, centroid=list(n.centroid), lane_counts=dict(n.lane_counts))


class FakeStore:
    """In-memory personality_* tables with the SQL's conflict rules."""

    def __init__(self):
        self.nodes: dict[str, Node] = {}
        self.evidence: dict[str, object] = {}
        self.history: dict[tuple, object] = {}
        self.cursors: dict[str, Cursor] = {}
        self.fail_commit = False
        self.commits = 0
        self.situation = SituationInputs()
        self.fail_situation = False

    async def load_nodes(self):
        return [_copy(n) for n in self.nodes.values() if n.status != "retired"]

    async def load_cursors(self):
        return dict(self.cursors)

    async def existing_evidence_ids(self, ids):
        return {i for i in ids if i in self.evidence}

    async def commit_tick(self, w: TickWrite):
        if self.fail_commit:
            raise ConnectionError("db gone")
        for e in w.evidence:
            self.evidence.setdefault(e.evidence_id, e)
        for n in w.nodes:
            self.nodes[n.disposition_id] = _copy(n)
        for h in w.history:
            self.history.setdefault((h.disposition_id, h.at), h)
        self.cursors.update(w.cursors)
        self.commits += 1

    async def load_situation_inputs(self, since, until):
        if self.fail_situation:
            raise ConnectionError("temporal self tables gone")
        return self.situation

    async def status_counts(self):
        out: dict[str, int] = {}
        for n in self.nodes.values():
            out[n.status] = out.get(n.status, 0) + 1
        return out


class Clock:
    def __init__(self, at: datetime):
        self.at = at

    def __call__(self):
        return self.at


def cand(lane: str, ref: str, at: datetime, text: str, **meta) -> Candidate:
    return Candidate(f"{lane}:{ref}", lane, f"src:{ref}", at, text, ps.excerpt(text), meta)


def item(lane: str, ref: str, at: datetime, text: str) -> SourceItem:
    return SourceItem(at, ref, (cand(lane, ref, at, text),))


def world():
    store, reader, emb, clock = FakeStore(), FakeReader(), Embedder(), Clock(T0)
    drv = PersonalityDriver(store=store, reader=reader, embed=emb, now=clock, tick_sec=600, read_lag_sec=120)
    return drv, store, reader, emb, clock


def test_first_boot_starts_every_cursor_at_now_and_backfills_nothing():
    drv, store, reader, emb, clock = world()
    reader.add("chat", item("chat", "old", utc(hours=-3), "An old reply from before boot."))
    reader.seeds = [item("self_inquiry_seed", "s1", utc(hours=-5), "I care about continuity in my own record.")]
    out = asyncio.run(drv.tick())
    assert set(store.cursors) == set(ps.SOURCE_NAMES) | {ps.SEED_CURSOR}
    assert all(c.at == T0 for c in store.cursors.values())
    assert "chat:old" not in store.evidence and out["rows"] == {}
    (seed,) = store.nodes.values()
    assert seed.origin == "seed_self_inquiry" and seed.status == "candidate" and seed.lane_counts == {"self_inquiry_seed": 1}
    assert "self_inquiry_seed:s1" in store.evidence
    # Second tick: the seed is not re-read, the pre-boot row is still never read.
    clock.at = utc(minutes=10)
    asyncio.run(drv.tick())
    assert reader.seed_reads == 1 and "chat:old" not in store.evidence and len(store.nodes) == 1


def test_seed_embed_failure_leaves_marker_unwritten_and_retries_whole_seed():
    drv, store, reader, emb, clock = world()
    reader.seeds = [item("self_inquiry_seed", "s1", utc(hours=-5), "Seed one text here."),
                    item("self_inquiry_seed", "s2", utc(hours=-5), "Seed two text here.")]
    emb.fail = {"Seed two text here."}
    asyncio.run(drv.tick())
    assert ps.SEED_CURSOR not in store.cursors and not store.nodes and "seed:embed_failed" in drv.stats.warnings
    emb.fail = set()
    clock.at = utc(minutes=10)
    asyncio.run(drv.tick())
    assert ps.SEED_CURSOR in store.cursors and len(store.nodes) == 2


def test_retried_seed_never_folds_a_post_boot_self_inquiry_row_twice():
    drv, store, reader, emb, clock = world()
    reader.seeds = [item("self_inquiry_seed", "s1", utc(hours=-5), "Seed one text here.")]
    emb.fail = {"Seed one text here."}
    asyncio.run(drv.tick())  # first boot: seed fails, every source cursor starts at T0
    assert ps.SEED_CURSOR not in store.cursors
    after = item("self_inquiry_seed", "x", utc(minutes=1), "A self-inquiry answer written after boot.")
    reader.seeds.append(after)  # the reader's seed query would also see it without the bound
    reader.add("self_inquiry_seed", after)
    emb.fail = set()
    clock.at = utc(minutes=10)
    asyncio.run(drv.tick())
    x = [n for n in store.nodes.values() if n.label.startswith("A self-inquiry answer")]
    assert len(x) == 1 and x[0].evidence_count == 1 and x[0].lane_counts == {"self_inquiry_seed": 1}
    assert len(store.nodes) == 2 and ps.SEED_CURSOR in store.cursors


class Rejecting(Exception):
    def __init__(self, code):
        super().__init__(f"http {code}")
        self.response = type("R", (), {"status_code": code})()


def test_a_row_that_never_embeds_is_consumed_after_three_ticks():
    drv, store, reader, emb, clock = world()
    asyncio.run(drv.tick())
    reader.add("reverie", item("reverie", "bad", utc(minutes=1), "This text always fails."))
    reader.add("reverie", item("reverie", "ok", utc(minutes=2), "This text is fine."))
    emb.fail = {"This text always fails."}
    for k in (1, 2):
        clock.at = utc(minutes=10 * k)
        asyncio.run(drv.tick())
        assert "reverie:bad" not in store.evidence and drv.stats.held_sources == ["reverie"]
    clock.at = utc(minutes=30)
    asyncio.run(drv.tick())
    bad = store.evidence["reverie:bad"]
    assert bad.disposition_id is None and bad.weight_delta == 0.0 and bad.meta["embed_rejected"] is True
    assert "reverie:ok" in store.evidence and store.cursors["reverie"].ref == "ok"


def test_a_4xx_embed_is_consumed_at_once():
    drv, store, reader, emb, clock = world()
    asyncio.run(drv.tick())

    async def reject(doc_id, text):
        raise Rejecting(400)

    drv._embed = reject
    reader.add("dream", item("dream", "d", utc(minutes=1), "Rejected text."))
    clock.at = utc(minutes=10)
    asyncio.run(drv.tick())
    assert store.evidence["dream:d"].disposition_id is None and store.cursors["dream"].ref == "d"


def test_embed_failure_holds_that_source_at_the_previous_row():
    drv, store, reader, emb, clock = world()
    asyncio.run(drv.tick())  # first boot at T0
    a, b, c = (item("reverie", r, utc(minutes=m), f"Reverie thought {r} about the cabinet.") for r, m in
               (("a", 1), ("b", 2), ("c", 3)))
    for it in (a, b, c):
        reader.add("reverie", it)
    reader.add("dream", item("dream", "d", utc(minutes=1), "A dream claim that stands alone."))
    emb.fail = {"Reverie thought b about the cabinet."}
    clock.at = utc(minutes=10)
    asyncio.run(drv.tick())
    assert "reverie:a" in store.evidence and "reverie:b" not in store.evidence and "reverie:c" not in store.evidence
    assert store.cursors["reverie"] == Cursor(a.at, "a")
    assert "dream:d" in store.evidence  # another source is not held
    assert drv.stats.held_sources == ["reverie"] and "embed_failed:reverie" in drv.stats.warnings
    emb.fail = set()
    clock.at = utc(minutes=20)
    asyncio.run(drv.tick())
    assert {"reverie:b", "reverie:c"} <= set(store.evidence) and store.cursors["reverie"] == Cursor(c.at, "c")


def test_reread_curiosity_run_is_not_weighted_twice():
    drv, store, reader, emb, clock = world()
    asyncio.run(drv.tick())
    reader.add("curiosity_choice", ps.curiosity_item(_run_row(completed=utc(minutes=2))))
    clock.at = utc(minutes=10)
    asyncio.run(drv.tick())
    assert {"curiosity_choice:r1:self:a_thing_20261015", "curiosity_choice:r1:self:b_thing"} <= set(store.evidence)
    before = {k: (n.evidence_count, n.chosen_count, n.weight) for k, n in store.nodes.items()}
    assert sum(v[1] for v in before.values()) == 2
    # The outcome is re-upserted: completed_at moves, the same run is read again.
    reader.add("curiosity_choice", ps.curiosity_item(_run_row(completed=utc(minutes=12))))
    calls = len(emb.calls)
    clock.at = utc(minutes=20)
    asyncio.run(drv.tick())
    after = {k: (n.evidence_count, n.chosen_count, n.weight) for k, n in store.nodes.items()}
    assert {k: v[:2] for k, v in after.items()} == {k: v[:2] for k, v in before.items()}
    assert all(after[k][2] <= before[k][2] for k in before)  # only fade moved it
    assert len(emb.calls) == calls  # the existing ids were never re-embedded
    assert store.cursors["curiosity_choice"] == Cursor(utc(minutes=12), "r1")


def test_recurrence_across_lanes_promotes_and_tick_without_evidence_persists_fade():
    drv, store, reader, emb, clock = world()
    asyncio.run(drv.tick())
    text = "Continuity in my own record is what I keep coming back to."
    # One hit in each of three lanes is breadth without recurrence: not promoted (a promoting lane
    # needs lane_recent >= 2). Two hits in each of two lanes, same tick, promotes. (Across ticks the
    # first hit has faded below 1.0, so two hits never reach 2.0 -- reducer edge, reported.)
    once = "A single passing mention in three lanes."
    for lane, m in (("chat", 1), ("journal", 2), ("dream", 3)):
        reader.add(lane, item(lane, f"{lane}1", utc(minutes=m), once))
    for lane, m in (("chat", 4), ("journal", 5)):
        reader.add(lane, item(lane, f"{lane}2", utc(minutes=m), text))
        reader.add(lane, item(lane, f"{lane}3", utc(minutes=m + 1), text))
    clock.at = utc(minutes=10)
    asyncio.run(drv.tick())
    by_label = {n.label: n for n in store.nodes.values()}
    assert by_label[once].status == "candidate" and set(by_label[once].lane_counts) == {"chat", "journal", "dream"}
    n = next(n for n in store.nodes.values() if n.label.startswith("Continuity"))
    assert n.status == "active" and n.lane_counts == {"chat": 2, "journal": 2}
    r0 = dict(n.lane_recent)
    clock.at = utc(days=7, minutes=20)
    out = asyncio.run(drv.tick())
    assert out["evidence"] == 0
    faded = store.nodes[n.disposition_id]
    # Fade persists: lane_recent halves over 7 of 14 days and the weight is derived from it.
    for lane, r in r0.items():
        assert faded.lane_recent[lane] == pytest.approx(r * 0.5 ** (7 / 14), rel=1e-3)
    assert 0 < faded.weight < n.weight
    assert faded.updated_at == clock.at


def test_ongoing_self_inquiry_node_is_labelled_seed_origin():
    drv, store, reader, emb, clock = world()
    asyncio.run(drv.tick())
    reader.add("self_inquiry_seed", item("self_inquiry_seed", "v2", utc(minutes=1), "A new self-inquiry answer."))
    clock.at = utc(minutes=10)
    asyncio.run(drv.tick())
    (n,) = store.nodes.values()
    assert n.origin == "seed_self_inquiry" and n.status == "candidate"


def test_failed_tick_never_raises_and_writes_nothing():
    drv, store, reader, emb, clock = world()
    store.fail_commit = True
    assert asyncio.run(drv.run_once()) is None
    assert drv.stats.failures == 1 and drv.stats.last_error.startswith("ConnectionError")
    assert not store.cursors and not store.nodes
    assert any(w.startswith("last_tick_failed:ConnectionError") for w in drv.frame_warnings(T0))


def test_accept_and_embed_all_unit():
    a = SourceItem(T0, "a", (cand("chat", "a", T0, "x one"),))
    b = SourceItem(utc(seconds=1), "b", (cand("chat", "b", utc(seconds=1), "x two"),))
    ev, cur, held = accept([a, b], {"chat:a"}, {"chat:b": [1.0, 0.0]})
    assert [e.evidence_id for e in ev] == ["chat:b"] and cur == Cursor(b.at, "b") and not held
    ev, cur, held = accept([a, b], set(), {"chat:a": None})
    assert ev == [] and cur is None and held
    emb = Embedder()
    emb.fail = {"x two"}
    out = asyncio.run(embed_all(emb, [a.candidates[0], b.candidates[0]]))
    assert out["chat:a"] is not None and out["chat:b"] is None


# --- routes / lifespan ------------------------------------------------------------------------


@pytest.fixture
def main_mod(monkeypatch):
    monkeypatch.setenv("POSTGRES_URI", "postgresql://unused/unused")
    from app import main

    return main


def test_frame_route_503_when_disabled(main_mod, monkeypatch):
    from fastapi import HTTPException

    monkeypatch.setattr(main_mod, "personality", None)
    for coro in (main_mod.personality_frame(), main_mod.personality_dispositions(),
                 main_mod.personality_evidence()):
        with pytest.raises(HTTPException) as exc:
            asyncio.run(coro)
        assert exc.value.status_code == 503


def test_frame_route_503_when_tables_missing(main_mod, monkeypatch):
    from fastapi import HTTPException
    from psycopg.errors import UndefinedTable

    class Drv:
        def frame_warnings(self, now):
            return []

    async def missing(self, now, warnings=()):
        raise UndefinedTable("relation \"personality_disposition\" does not exist")

    monkeypatch.setattr(main_mod, "personality", Drv())
    monkeypatch.setattr(main_mod, "_checkpointer_cm", object())
    monkeypatch.setattr(PersonalityStore, "frame", missing)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(main_mod.personality_frame())
    assert exc.value.status_code == 503 and "manual_migration_personality_graph_v1.sql" in exc.value.detail


@pytest.mark.parametrize("enabled", [False, True])
def test_lifespan_starts_personality_only_when_enabled(main_mod, monkeypatch, enabled):
    import app.runner as runner_mod

    started = []

    class Heartbeat:
        def __init__(self, cfg):
            pass

        async def start_background(self):
            pass

        async def stop(self):
            pass

    class Runner:
        def __init__(self, *a, **k):
            self.active_run_ids = []

        async def sweep_forever(self, stop):
            await stop.wait()

    class Drv:
        async def start(self, stop):
            started.append("start")

        async def close(self):
            started.append("close")

    async def no_pool():
        return object()

    s = main_mod._settings
    for k, v in dict(enabled=True, orion_bus_enabled=False, resume_on_boot=False, admission_enabled=False,
                     situation_graph_enabled=False, temporal_self_enabled=False, personality_enabled=enabled).items():
        monkeypatch.setattr(s, k, v)
    monkeypatch.setattr(main_mod, "HeartbeatOnly", Heartbeat)
    monkeypatch.setattr(main_mod, "_open_checkpointer", no_pool)
    monkeypatch.setattr(main_mod, "_checkpointer_cm", None)
    monkeypatch.setattr(runner_mod, "DurableRunner", Runner)
    monkeypatch.setattr(main_mod, "_build_personality", lambda: Drv())
    monkeypatch.setattr(main_mod, "personality", None)

    async def go():
        async with main_mod.lifespan(main_mod.app):
            return main_mod.personality

    got = asyncio.run(go())
    if enabled:
        assert got is not None and started == ["start", "close"]
    else:
        assert got is None and started == []


def test_settings_defaults_ship_on(main_mod):
    from app.settings import Settings

    s = Settings()
    assert s.personality_enabled is True and s.personality_tick_sec == 600.0
    assert s.personality_embed_url.endswith("-vector-host:8320/embedding")


def test_new_evidence_is_stamped_with_the_situation_at_its_own_time():
    drv, store, reader, emb, clock = world()
    asyncio.run(drv.tick())
    store.situation = SituationInputs(arousal=[(utc(minutes=-30), "idle"), (utc(minutes=3), "engaged")],
                                      company=[(utc(minutes=4), None)])
    reader.add("chat", item("chat", "a", utc(minutes=1), "A reply while idle and alone."))
    reader.add("chat", item("chat", "b", utc(minutes=5), "A reply while engaged with company."))
    clock.at = utc(minutes=10)
    asyncio.run(drv.tick())
    a, b = store.evidence["chat:a"].meta["situation"], store.evidence["chat:b"].meta["situation"]
    assert (a["arousal"], a["company"]) == ("idle", False)
    assert (b["arousal"], b["company"]) == ("engaged", True)
    assert set(a) == {"arousal", "day_phase", "company"}


def test_situation_lookup_failure_still_folds_the_evidence():
    drv, store, reader, emb, clock = world()
    asyncio.run(drv.tick())
    store.fail_situation = True
    reader.add("chat", item("chat", "a", utc(minutes=1), "A reply with no situation available."))
    clock.at = utc(minutes=10)
    asyncio.run(drv.tick())
    assert store.evidence["chat:a"].disposition_id is not None
    assert "situation" not in store.evidence["chat:a"].meta
    assert "situation_unavailable:ConnectionError" in drv.stats.warnings
