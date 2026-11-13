"""Personality graph loop, rev 4 (choices only): store transaction, curiosity sync, bus intake,
driver tick, routes. No database here.

What is pinned: the curiosity sync backfills every stored curiosity choice on its first tick and is
idempotent (a re-read run inserts nothing, folds nothing twice); a bus choice is inserted once and
folded on the next tick; each choice becomes ONE evidence row whose direction is chosen minus mean
declined; a failed embed holds the choice cursor at the previous row and gives up after three ticks;
a tick with no evidence still persists fade; rev 2 prose nodes are never loaded; a failing tick never
raises; the routes answer 503 when the loop is disabled or its tables are missing.
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

from orion.personality.graph import Node, choice_direction  # noqa: E402
from orion.personality.situation import SituationInputs  # noqa: E402
from orion.schemas.personality import ChoiceOptionV1, PersonalityChoiceV1  # noqa: E402

from app import personality_sources as ps  # noqa: E402
from app.personality_driver import PersonalityDriver, choice_evidence, option_texts  # noqa: E402
from app.personality_sources import Cursor  # noqa: E402
from app.personality_store import (  # noqa: E402
    INSERT_CHOICE_SQL,
    INSERT_EVIDENCE_SQL,
    INSERT_HISTORY_SQL,
    UPDATE_NODE_WEIGHT_SQL,
    UPSERT_CURSOR_SQL,
    UPSERT_NODE_SQL,
    PersonalityStore,
    TickWrite,
    choice_row,
    node_key,
    split_nodes,
)

T0 = datetime(2026, 11, 13, 18, 0, tzinfo=timezone.utc)


def utc(**kw) -> datetime:
    return T0 + timedelta(**kw)


def opt(key: str, rank=None, text=None) -> ChoiceOptionV1:
    return ChoiceOptionV1(key=key, text=text or key, market_rank=rank)


def choice(cid: str, at: datetime, *, chosen="people", declined=("pipeline",), rank=2, lane="prime",
           declined_all=False, **kw) -> PersonalityChoiceV1:
    alts = [opt(t, i + (1 if rank != 1 else 2)) for i, t in enumerate(declined)]
    if declined_all:
        return PersonalityChoiceV1(choice_id=cid, lane=lane, occurred_at=at, declined_all=True, alternatives=alts, **kw)
    return PersonalityChoiceV1(choice_id=cid, lane=lane, occurred_at=at, chosen=opt(chosen, rank), alternatives=alts,
                               chosen_market_rank=rank, act_id=kw.pop("act_id", f"act:{cid}"), **kw)


def direction_of(chosen: str, declined) -> list[float]:
    return choice_direction(vec_for(chosen), [vec_for(t) for t in declined])


# --- sources + choice -> evidence (pure) --------------------------------------------------------


def test_sql_is_keyset_capped_and_reads_only_choices():
    for sql in (ps.CURIOSITY_SQL, ps.CHOICE_SQL, ps.NAMING_SQL):
        assert "LIMIT %(limit)s" in sql and "%(hi)s" in sql and "%(ref)s" in sql
    assert "FROM personality_choice" in ps.CHOICE_SQL and "ORDER BY created_at, choice_id" in ps.CHOICE_SQL
    assert "d.arm <> 'self_inquiry'" in ps.CURIOSITY_SQL
    # Rev 2 prose sources are gone (kill means kill).
    for gone in ("SOURCES", "SEED_SQL", "reverie_item", "journal_item", "excerpt"):
        assert not hasattr(ps, gone)


def test_choice_evidence_direction_weight_label():
    c = choice("prime:t1", T0, chosen="people", declined=("pipeline", "heat"), rank=3)
    vecs = {t: vec_for(t) for t in ("people", "pipeline", "heat")}
    e = choice_evidence(c, vecs)
    assert e.evidence_id == "choice:prime:t1" and e.lane == "prime" and e.source_ref == "act:prime:t1"
    assert e.weight == 1.0 and e.excerpt == "people over pipeline (+1 more)"
    assert e.embedding == pytest.approx(direction_of("people", ("pipeline", "heat")))
    assert e.meta["chosen_market_rank"] == 3 and e.meta["alternatives"] == 2
    top = choice_evidence(choice("prime:t2", T0, rank=1), {t: vec_for(t) for t in ("people", "pipeline")})
    assert top.weight == 0.25


def test_declined_all_reads_as_a_decline_of_every_option():
    c = choice("outreach:o1", T0, lane="outreach", declined=("topic a", "topic b"), declined_all=True)
    assert option_texts(c) == ("say nothing", ["topic a", "topic b"])
    e = choice_evidence(c, {t: vec_for(t) for t in ("say nothing", "topic a", "topic b")})
    assert e.excerpt == "say nothing over topic a (+1 more)" and e.weight == 1.0 and e.meta["declined_all"] is True
    assert option_texts(choice("prime:x", T0, declined_all=True))[0] == "no move"


def test_missing_vector_gives_an_empty_direction():
    c = choice("prime:t1", T0)
    assert choice_evidence(c, {"people": vec_for("people")}).embedding == []


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


def _after(key, cur):
    return cur is None or key > (cur.at, cur.ref)


class FakeReader:
    def __init__(self, store):
        self.store = store
        self.curiosity: list[dict] = []  # joined curiosity rows
        self.namings: list[dict] = []  # orion_disposition_naming rows

    async def read_curiosity(self, cursor, hi):
        # Mirrors CURIOSITY_SQL's NOT EXISTS: a run already holding a row is never read again.
        synced = {r["act_id"] for r in self.store.choices.values()}
        rows = sorted(self.curiosity, key=lambda r: (r["completed_at"], r["run_id"]))
        return [r for r in rows if _after((r["completed_at"], r["run_id"]), cursor) and r["completed_at"] < hi
                and f"curiosity_run:{r['run_id']}" not in synced][:200]

    async def read_choices(self, cursor, hi):
        rows = sorted(self.store.choices.values(), key=lambda r: (r["created_at"], r["choice_id"]))
        return [dict(r) for r in rows if _after((r["created_at"], r["choice_id"]), cursor) and r["created_at"] < hi][:200]

    async def read_namings(self, cursor, hi):
        rows = sorted(self.namings, key=lambda r: (r["created_at"], r["naming_id"]))
        return [r for r in rows if _after((r["created_at"], r["naming_id"]), cursor) and r["created_at"] < hi][:200]


def _copy(n: Node) -> Node:
    return replace(n, centroid=list(n.centroid), lane_counts=dict(n.lane_counts), lane_recent=dict(n.lane_recent))


class FakeStore:
    """In-memory personality_* tables with the SQL's conflict rules and origin filter."""

    def __init__(self, clock):
        self.clock = clock
        self.nodes: dict[str, Node] = {}
        self.evidence: dict[str, object] = {}
        self.history: dict[tuple, object] = {}
        self.cursors: dict[str, Cursor] = {}
        self.choices: dict[str, dict] = {}
        self.fail_commit = False
        self.commits = 0
        self.situation = SituationInputs()
        self.watermark = None  # None = "no chronicle" (company unknown, warned)
        self.fail_situation = False

    async def load_nodes(self):
        return [_copy(n) for n in self.nodes.values() if n.status != "retired" and n.origin == "choice"]

    async def load_cursors(self):
        return dict(self.cursors)

    async def existing_evidence_ids(self, ids):
        return {i for i in ids if i in self.evidence}

    async def insert_choices(self, choices, *, written_by, cursor=None):
        new = 0
        for c in choices:
            if c.choice_id not in self.choices:
                self.choices[c.choice_id] = {**choice_row(c, written_by), "created_at": self.clock()}
                new += 1
        if cursor is not None:
            self.cursors[cursor[0]] = cursor[1]
        return new

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

    async def chronicle_watermark(self):
        return self.watermark

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


def world():
    clock = Clock(T0)
    store = FakeStore(clock)
    reader, emb = FakeReader(store), Embedder()
    drv = PersonalityDriver(store=store, reader=reader, embed=emb, now=clock, tick_sec=600, read_lag_sec=120)
    return drv, store, reader, emb, clock


def publish(drv, c: PersonalityChoiceV1) -> str:
    """What the bus handler does with one orion:personality:choice payload."""
    return asyncio.run(drv.offer_choice(c.model_dump(mode="json")))


def _run_row(run_id="r1", completed=T0, tested=("self:a_thing_20261015", "self:b_thing")):
    return {"run_id": run_id, "arm": "value_order", "turn_started_at": completed - timedelta(minutes=30),
            "decided_at": completed - timedelta(minutes=31), "completed_at": completed, "turn_ok": True,
            "offered": [{"rank": 1, "prior_id": "self:a_thing_20261015"}, {"rank": 2, "prior_id": "self:c_thing"},
                        {"rank": 3, "prior_id": "self:d_thing"}],
            "stale_offered": [{"rank": 1, "prior_id": "self:b_thing"}],
            "per_prior": [{"kind": "tested", "prior_id": p} for p in tested]}


def test_curiosity_backfill_runs_on_the_first_tick_and_is_idempotent():
    drv, store, reader, emb, clock = world()
    reader.curiosity = [_run_row("r1", utc(days=-20)), _run_row("r2", utc(days=-3), tested=("self:c_thing",))]
    out = asyncio.run(drv.tick())
    assert out["curiosity_synced"] == 3 and set(store.choices) == {
        "curiosity:r1:self:a_thing_20261015", "curiosity:r1:self:b_thing", "curiosity:r2:self:c_thing"}
    assert {r["written_by"] for r in store.choices.values()} == {"curiosity_sync"}
    assert store.cursors["curiosity_sync"] == Cursor(utc(days=-3), "r2")
    assert out["evidence"] == 0  # inserted at T0: inside the read lag, folded next tick
    clock.at = utc(minutes=10)
    out = asyncio.run(drv.tick())
    assert out["evidence"] == 3 and out["curiosity_synced"] == 0
    ev = store.evidence["choice:curiosity:r1:self:a_thing_20261015"]
    # rank 1 of a two-pick run, 20 days old: 0.25 / 2, faded by 20 of 14 days' half-life.
    age_days = (20 * 86400 + 40 * 60) / 86400  # occurred at the turn start (30 min before completion), read 10 min after T0
    assert ev.weight_delta == pytest.approx(0.125 * 0.5 ** (age_days / 14), rel=1e-6)
    assert all(n.origin == "choice" for n in store.nodes.values())
    # The outcome is re-upserted with a DIFFERENT tested set (Hub replaces per_prior): the run is
    # already synced, so nothing new is inserted or folded -- one run stays one decision.
    reader.curiosity.append(_run_row("r1", utc(minutes=11), tested=("self:c_thing", "self:d_thing")))
    calls = len(emb.calls)
    clock.at = utc(minutes=20)
    out = asyncio.run(drv.tick())
    assert out["curiosity_synced"] == 0 and out["evidence"] == 0 and len(store.choices) == 3
    assert not any("c_thing" in k for k in store.choices if k.startswith("curiosity:r1:"))
    assert len(emb.calls) == calls
    assert "curiosity_run:' || o.run_id" in ps.CURIOSITY_SQL and "NOT EXISTS" in ps.CURIOSITY_SQL


def test_bus_choice_is_inserted_once_and_folded_on_the_next_tick():
    drv, store, reader, emb, clock = world()
    asyncio.run(drv.tick())
    c = choice("prime:t1", utc(minutes=1), rank=3)
    assert publish(drv, c) == "inserted" and publish(drv, c) == "duplicate"
    assert asyncio.run(drv.offer_choice({"choice_id": "x", "lane": "prime"})) == "invalid"
    # The curiosity lane is durable-runs' own sync: a bus producer may not write it.
    assert publish(drv, choice("curiosity:r9:p", utc(minutes=1), lane="curiosity")) == "invalid"
    # A choice dated in the future would never fade.
    assert publish(drv, choice("prime:future", utc(hours=1))) == "invalid"
    assert drv.stats.bus_choices_inserted == 1 and drv.stats.bus_choices_invalid == 3
    assert store.choices["prime:t1"]["written_by"] == "bus"
    clock.at = utc(minutes=10)
    asyncio.run(drv.tick())
    ev = store.evidence["choice:prime:t1"]
    (n,) = store.nodes.values()
    assert ev.disposition_id == n.disposition_id and ev.lane == "prime" and ev.weight_delta == pytest.approx(1.0, rel=1e-3)
    assert n.label == "people over pipeline, 1 of 1" and n.origin == "choice" and n.status == "candidate"
    assert ev.meta["choice_id"] == "prime:t1" and ev.meta["act_id"] == "act:prime:t1"
    assert store.cursors["choice"].ref == "prime:t1"


def test_repeated_departures_promote_a_tendency():
    drv, store, reader, emb, clock = world()
    for i in range(4):  # 4 x 1.0, faded a few minutes: past the 3.0 promotion line
        publish(drv, choice(f"prime:t{i}", utc(minutes=1), rank=2))
    clock.at = utc(minutes=10)
    asyncio.run(drv.tick())
    (n,) = store.nodes.values()
    assert n.status == "active" and n.evidence_count == 4 and n.label.endswith(", 4 of 4")


def test_prose_nodes_are_never_loaded_or_touched():
    drv, store, reader, emb, clock = world()
    prose = Node(disposition_id="disp-old", origin="recurrence", status="active", label="The perception error is high",
                 label_similarity=0.9, centroid=vec_for("x"), centroid_n=9, weight=5.0, evidence_count=9,
                 lane_counts={"reverie": 9}, chosen_count=0, first_seen_at=utc(days=-9), last_reinforced_at=utc(days=-1),
                 retired_at=None, updated_at=utc(days=-1), lane_recent={"substrate_narration": 4.0})
    store.nodes["disp-old"] = prose
    publish(drv, choice("prime:t1", utc(minutes=1)))
    clock.at = utc(minutes=10)
    asyncio.run(drv.tick())
    assert store.nodes["disp-old"] == prose
    assert store.evidence["choice:prime:t1"].disposition_id != "disp-old"


def test_tick_without_evidence_persists_fade():
    drv, store, reader, emb, clock = world()
    for i in range(3):
        publish(drv, choice(f"prime:t{i}", utc(minutes=1)))
    clock.at = utc(minutes=10)
    asyncio.run(drv.tick())
    (n,) = store.nodes.values()
    clock.at = utc(days=7, minutes=10)
    out = asyncio.run(drv.tick())
    assert out["evidence"] == 0
    faded = store.nodes[n.disposition_id]
    assert faded.weight == pytest.approx(n.weight * 0.5 ** (7 / 14), rel=1e-3) and faded.updated_at == clock.at


class Rejecting(Exception):
    def __init__(self, code):
        super().__init__(f"http {code}")
        self.response = type("R", (), {"status_code": code})()


def test_embed_outage_holds_the_cursor_and_never_consumes_a_choice():
    drv, store, reader, emb, clock = world()
    publish(drv, choice("prime:a", utc(minutes=1), chosen="ok one"))
    clock.at = utc(seconds=61)
    publish(drv, choice("prime:b", utc(minutes=1), chosen="always fails"))
    clock.at = utc(seconds=62)
    publish(drv, choice("prime:c", utc(minutes=1), chosen="ok two"))
    emb.fail = {"always fails"}
    for k in range(1, 7):  # an hour of outage
        clock.at = utc(minutes=10 * k)
        asyncio.run(drv.tick())
        assert "choice:prime:a" in store.evidence and "choice:prime:b" not in store.evidence
        assert "choice:prime:c" not in store.evidence and store.cursors["choice"].ref == "prime:a"
        assert drv.stats.held and "embed_failed:choice:prime:b" in drv.stats.warnings
    emb.fail = set()  # the embedder is back
    clock.at = utc(minutes=70)
    asyncio.run(drv.tick())
    assert store.evidence["choice:prime:b"].disposition_id is not None
    assert "choice:prime:c" in store.evidence and store.cursors["choice"].ref == "prime:c" and not drv.stats.held


def test_a_4xx_embed_is_consumed_at_once():
    drv, store, reader, emb, clock = world()

    async def reject(doc_id, text):
        raise Rejecting(400)

    drv._embed = reject
    publish(drv, choice("prime:a", utc(minutes=1)))
    clock.at = utc(minutes=10)
    asyncio.run(drv.tick())
    assert store.evidence["choice:prime:a"].disposition_id is None and store.cursors["choice"].ref == "prime:a"


def test_option_vectors_are_cached_across_ticks():
    drv, store, reader, emb, clock = world()
    publish(drv, choice("prime:a", utc(minutes=1)))
    clock.at = utc(minutes=10)
    asyncio.run(drv.tick())
    publish(drv, choice("prime:b", utc(minutes=11)))
    clock.at = utc(minutes=20)
    asyncio.run(drv.tick())
    assert sorted(emb.calls) == ["people", "pipeline"]


def test_an_invalid_stored_row_is_consumed_as_skipped():
    drv, store, reader, emb, clock = world()
    publish(drv, choice("prime:a", utc(minutes=1)))
    store.choices["prime:a"]["alternatives"] = "[]"
    clock.at = utc(minutes=10)
    asyncio.run(drv.tick())
    row = store.evidence["choice:prime:a"]
    assert row.disposition_id is None and row.meta["skip_reason"] == "unknown_lane"
    assert store.cursors["choice"].ref == "prime:a"


def test_failed_tick_never_raises_and_writes_nothing():
    drv, store, reader, emb, clock = world()
    publish(drv, choice("prime:a", utc(minutes=-10)))
    clock.at = utc(minutes=10)
    store.fail_commit = True
    assert asyncio.run(drv.run_once()) is None
    assert drv.stats.failures == 1 and drv.stats.last_error.startswith("ConnectionError")
    assert "choice" not in store.cursors and not store.nodes
    assert any(w.startswith("last_tick_failed:ConnectionError") for w in drv.frame_warnings(utc(minutes=10)))


def test_health_reports_choice_intake():
    drv, store, reader, emb, clock = world()
    publish(drv, choice("prime:a", utc(minutes=1)))
    h = drv.health()
    assert h["bus_choices_inserted"] == 1 and "choices_read_last_tick" in h and "curiosity_synced_last_tick" in h

# --- store (fake connection) ------------------------------------------------------------------


class FakeCursor:
    rowcount = 1

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
    return Node(disposition_id=did, origin="choice", status=status, label=label, label_similarity=1.0,
                centroid=[1.0, 0.0], centroid_n=centroid_n, weight=weight, evidence_count=centroid_n,
                lane_counts=lanes or {"chat": 1}, chosen_count=0, first_seen_at=T0, last_reinforced_at=T0,
                retired_at=None, updated_at=updated)


def test_commit_tick_is_one_transaction_in_migration_order():
    from orion.personality.graph import EvidenceRow, HistoryRow

    pool = FakePool()
    ev = EvidenceRow("choice:prime:c1", "prime", "prime_thought:c1", T0, "x over y", "d2", 1.0, 1.0, {"a": 1})
    w = TickWrite(now=T0, evidence=[ev], nodes=[node("d1", weight=0.9), node("d2", centroid_n=1)],
                  loaded={"d1": node_key(node("d1"))}, history=[HistoryRow("d2", T0, 1.0, "candidate", 1)],
                  cursors={"choice": Cursor(T0, "c1"), "naming": Cursor(T0, "")})
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
    assert curs["choice"]["ref"] == "c1" and curs["naming"]["ref"] is None


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
    full, light = split_nodes([replace(a, lane_counts={"chat": 1, "prime": 1})], loaded)
    assert [r["disposition_id"] for r in full] == ["a"] and light == []
    # Opposition moves the label's counts: a full upsert too.
    full, light = split_nodes([replace(a, opposed_count=1)], loaded)
    assert [r["disposition_id"] for r in full] == ["a"] and light == []




def test_insert_choices_is_one_insert_only_transaction_with_its_cursor():
    pool = FakePool()
    c = choice("prime:t1", T0)
    n = asyncio.run(PersonalityStore(pool).insert_choices([c, choice("prime:t2", T0)], written_by="curiosity_sync",
                                                          cursor=("curiosity_sync", Cursor(T0, "r9"))))
    assert n == 2
    kinds = [e[0] for e in pool.log]
    assert kinds[0] == "begin" and kinds[-1] == "commit" and kinds.count("begin") == 1
    sqls = [e[1] for e in pool.log if e[0] == "execute"]
    assert sqls[1:] == [INSERT_CHOICE_SQL, INSERT_CHOICE_SQL, UPSERT_CURSOR_SQL]
    assert "ON CONFLICT (choice_id) DO NOTHING" in INSERT_CHOICE_SQL
    row = next(e[2] for e in pool.log if e[0] == "execute" and e[1] == INSERT_CHOICE_SQL)
    assert row["written_by"] == "curiosity_sync" and json.loads(row["chosen"])["key"] == "people"
    assert [o["key"] for o in json.loads(row["alternatives"])] == ["pipeline"]
    # A declined_all row stores chosen as SQL NULL.
    assert choice_row(choice("prime:o1", T0, declined_all=True), "bus")["chosen"] is None


def test_choice_row_round_trips_through_the_reader():
    c = choice("prime:t1", T0, situation={"arousal": "engaged"})
    row = {**choice_row(c, "bus"), "created_at": T0}
    assert ps.choice_from_row(row) == c
    assert ps.choice_from_row({**row, "alternatives": "[]"}) is None  # no longer a choice: skipped, not raised


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
    assert exc.value.status_code == 503 and "manual_migration_personality_choice_v1.sql" in exc.value.detail


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
    store.situation = SituationInputs(arousal=[(utc(minutes=-30), "idle"), (utc(minutes=3), "engaged")],
                                      company=[(utc(minutes=4), None)])
    publish(drv, choice("prime:a", utc(minutes=1), situation={"binding": "heat"}))
    publish(drv, choice("prime:b", utc(minutes=5)))
    clock.at = utc(minutes=10)
    asyncio.run(drv.tick())
    a, b = store.evidence["choice:prime:a"].meta["situation"], store.evidence["choice:prime:b"].meta["situation"]
    assert store.evidence["choice:prime:a"].meta["choice_situation"] == {"binding": "heat"}
    assert (a["arousal"], a["company"]) == ("idle", False)
    assert (b["arousal"], b["company"]) == ("engaged", True)
    assert set(a) == {"arousal", "day_phase", "company"}


def test_situation_lookup_failure_still_folds_the_evidence():
    drv, store, reader, emb, clock = world()
    store.fail_situation = True
    publish(drv, choice("prime:a", utc(minutes=1)))
    clock.at = utc(minutes=10)
    asyncio.run(drv.tick())
    assert store.evidence["choice:prime:a"].disposition_id is not None
    assert "situation" not in store.evidence["choice:prime:a"].meta
    assert "situation_unavailable:ConnectionError" in drv.stats.warnings


# --- situation mix (observability only) ---------------------------------------------------------


def test_situation_mix_folds_counts_and_skips_missing_keys():
    from app.personality_store import situation_mix_by_id

    rows = [
        {"disposition_id": "d1", "dim": "n", "val": "", "c": 3},
        {"disposition_id": "d1", "dim": "arousal", "val": "idle", "c": 2},
        {"disposition_id": "d1", "dim": "arousal", "val": None, "c": 1},  # a situation without arousal
        {"disposition_id": "d1", "dim": "day_phase", "val": "night", "c": 3},
        {"disposition_id": "d1", "dim": "company", "val": "true", "c": 1},
        {"disposition_id": "d1", "dim": "company", "val": "false", "c": 2},
        {"disposition_id": "d2", "dim": "n", "val": "", "c": 1},
        {"disposition_id": "d2", "dim": "company", "val": "maybe", "c": 1},  # not a bool: not counted
    ]
    out = situation_mix_by_id(rows)
    assert out["d1"] == {"n": 3, "arousal": {"idle": 2}, "day_phase": {"night": 3},
                         "company": {"true": 1, "false": 2}}
    assert out["d2"] == {"n": 1, "arousal": {}, "day_phase": {}, "company": {"true": 0, "false": 0}}


class ScriptedConn:
    """Answers each execute with the next scripted rows; logs (sql, params)."""

    def __init__(self, answers, log):
        self.answers, self.log = list(answers), log

    async def execute(self, sql, params=None):
        self.log.append((sql, params))
        rows = self.answers.pop(0)

        class R:
            async def fetchall(self_inner):
                return rows

        return R()


class ScriptedPool:
    def __init__(self, answers):
        self.answers, self.log = answers, []

    def connection(self):
        pool = self

        class _CM:
            async def __aenter__(self):
                return ScriptedConn(pool.answers, pool.log)

            async def __aexit__(self, *a):
                return False

        return _CM()


def test_dispositions_carry_situation_mix_from_one_query_bounded_to_returned_ids():
    from app.personality_store import SITUATION_MIX_SQL

    nodes = [{"disposition_id": "d1", "weight": 2.0}, {"disposition_id": "d2", "weight": 1.0}]
    mix = [{"disposition_id": "d1", "dim": "n", "val": "", "c": 2},
           {"disposition_id": "d1", "dim": "arousal", "val": "engaged", "c": 2}]
    pool = ScriptedPool([nodes, mix, []])
    rows = asyncio.run(PersonalityStore(pool).dispositions("all"))
    assert [sql for sql, _ in pool.log].count(SITUATION_MIX_SQL) == 1
    assert pool.log[1] == (SITUATION_MIX_SQL, {"ids": ["d1", "d2"]})
    assert rows[0]["situation_mix"]["n"] == 2 and rows[0]["situation_mix"]["arousal"] == {"engaged": 2}
    # A node with no stamped evidence still gets the full shape, n=0.
    assert rows[1]["situation_mix"] == {"n": 0, "arousal": {}, "day_phase": {}, "company": {"true": 0, "false": 0}}
    # No nodes -> no extra query.
    empty = ScriptedPool([[]])
    assert asyncio.run(PersonalityStore(empty).dispositions("active")) == [] and len(empty.log) == 1
    assert "jsonb_typeof(e.meta->'situation') = 'object'" in SITUATION_MIX_SQL


def at_clock(drv, clock, when, c):
    clock.at = when
    publish(drv, c)


def test_reads_wait_for_the_chronicle_so_company_is_final_when_stamped():
    drv, store, reader, emb, clock = world()
    store.watermark = utc(minutes=0)
    at_clock(drv, clock, utc(minutes=2), choice("prime:early", utc(minutes=2)))
    at_clock(drv, clock, utc(minutes=7), choice("prime:late", utc(minutes=7)))
    store.watermark = utc(minutes=6)  # read bound = watermark - COMPANY_CLOSE_GRACE (3 min) = 3
    clock.at = utc(minutes=10)
    asyncio.run(drv.tick())
    assert "choice:prime:early" in store.evidence and "choice:prime:late" not in store.evidence
    store.watermark = utc(minutes=12)
    clock.at = utc(minutes=20)
    asyncio.run(drv.tick())
    assert "choice:prime:late" in store.evidence
    assert not any(w.startswith("situation_chronicle_lagging") for w in drv.stats.warnings)


def test_a_stalled_chronicle_does_not_stall_personality():
    drv, store, reader, emb, clock = world()
    store.watermark = utc(minutes=-60)  # chronicle an hour behind
    publish(drv, choice("prime:a", utc(minutes=1)))
    clock.at = utc(minutes=10)
    asyncio.run(drv.tick())
    assert "choice:prime:a" in store.evidence
    assert "situation_chronicle_lagging:company_unknown_past_watermark" in drv.stats.warnings



def test_a_backdated_company_close_is_folded_before_rows_near_it_are_stamped():
    # Review 2026-10-17: the chronicle ends a company arc at the last sighting (recent row) and
    # learns of it only when the presence grace runs out (absent row, ~2 min later). A row read
    # inside that window used to be stamped "with company" for an arc that then closed earlier.
    drv, store, reader, emb, clock = world()
    store.watermark = utc(minutes=0)
    seen_last = utc(minutes=20)  # recent row: the person left here
    at_clock(drv, clock, utc(minutes=21), choice("prime:after_leaving", utc(minutes=21)))
    # Chronicle has folded up to 22:00 but not the absent row (due ~22:00+): arc still open.
    store.situation = SituationInputs(company=[(utc(minutes=-30), None)], company_known_until=utc(minutes=19))
    store.watermark = utc(minutes=22)
    clock.at = utc(minutes=24)
    asyncio.run(drv.tick())
    assert "choice:prime:after_leaving" not in store.evidence  # held: bound is 22 - 3 = 19
    # The absent row is folded: the arc closes, backdated to the last sighting.
    store.situation = SituationInputs(company=[(utc(minutes=-30), seen_last)], company_known_until=utc(minutes=27))
    store.watermark = utc(minutes=30)
    clock.at = utc(minutes=32)
    asyncio.run(drv.tick())
    assert store.evidence["choice:prime:after_leaving"].meta["situation"]["company"] is False


def test_covered_days_is_the_graph_age_clamped_to_the_window():
    from app.personality_store import covered_days

    born = utc(minutes=0)
    assert covered_days(born + timedelta(hours=3), born) == 1
    assert covered_days(born + timedelta(days=2, hours=1), born) == 3
    assert covered_days(born + timedelta(days=40), born) == 14
    assert covered_days(born, None) == 1


def test_choice_window_covers_the_oldest_choice_held():
    from app.personality_store import choice_window_days
    from orion.personality.graph import choice_style_from_rows

    rows = [{"choice_id": "c1", "act_id": "a1", "chosen_market_rank": 2, "declined_all": False,
             "occurred_at": utc(hours=-1, minutes=-51)}]
    for hours in (1, 21, 22.5, 23.9, 24.5, 47):
        now = T0 + timedelta(hours=hours)
        window = choice_window_days(now, None, [r["occurred_at"] for r in rows])
        style = choice_style_from_rows(rows, now=now, window_days=window)
        assert style is not None and style.choices == 1, hours


# --- bus intake (consumer) ----------------------------------------------------------------------


def _envelope(c: PersonalityChoiceV1):
    from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
    from orion.schemas.personality import PERSONALITY_CHOICE_KIND

    return BaseEnvelope(kind=PERSONALITY_CHOICE_KIND, source=ServiceRef(name="orion-mind", version="t", node="n"),
                        correlation_id="00000000-0000-0000-0000-000000000001", payload=c.model_dump(mode="json"))


def test_bus_handler_routes_a_choice_envelope_to_the_driver(main_mod, monkeypatch):
    got = []

    class Drv:
        async def offer_choice(self, payload):
            got.append(payload)
            return "inserted"

    monkeypatch.setattr(main_mod, "personality", Drv())
    c = choice("prime:t1", T0)
    asyncio.run(main_mod._handle_request(_envelope(c)))
    assert len(got) == 1 and PersonalityChoiceV1.model_validate(got[0]) == c


def test_bus_handler_drops_a_choice_when_the_loop_is_off_and_survives_a_db_error(main_mod, monkeypatch, caplog):
    monkeypatch.setattr(main_mod, "personality", None)
    asyncio.run(main_mod._handle_request(_envelope(choice("prime:t1", T0))))
    assert "personality_choice_dropped" in caplog.text

    class Broken:
        async def offer_choice(self, payload):
            raise ConnectionError("db gone")

    monkeypatch.setattr(main_mod, "personality", Broken())
    asyncio.run(main_mod._handle_request(_envelope(choice("prime:t2", T0))))  # never raises
    assert "personality_choice_write_failed" in caplog.text


@pytest.mark.parametrize("enabled", [False, True])
def test_choice_channel_is_subscribed_only_with_the_loop_on(main_mod, monkeypatch, enabled):
    import app.runner as runner_mod
    from orion.schemas.personality import PERSONALITY_CHOICE_CHANNEL

    patterns_seen = []

    class Quiet:
        def __init__(self, *a, **k):
            if "patterns" in k:
                patterns_seen.append(list(k["patterns"]))
            self.active_run_ids = []

        async def start_background(self):
            pass

        async def connect(self):
            pass

        def start(self):
            pass

        async def stop(self):
            pass

        async def close(self):
            pass

        async def sweep_forever(self, stop):
            await stop.wait()

    class Drv:
        async def start(self, stop):
            pass

        async def close(self):
            pass

    async def no_pool():
        return object()

    s = main_mod._settings
    for k, v in dict(enabled=True, orion_bus_enabled=True, resume_on_boot=False, admission_enabled=False,
                     situation_graph_enabled=False, temporal_self_enabled=False, physiology_bid_enabled=False,
                     cooling_act_enabled=False, checkpoint_prune_mode="off", personality_enabled=enabled).items():
        monkeypatch.setattr(s, k, v)
    for name in ("HeartbeatOnly", "OrionBusAsync", "Hunter"):
        monkeypatch.setattr(main_mod, name, Quiet)
    monkeypatch.setattr(main_mod, "build_rpc_health_publisher", lambda: Quiet())
    monkeypatch.setattr(main_mod, "_open_checkpointer", no_pool)
    monkeypatch.setattr(main_mod, "_checkpointer_cm", None)
    monkeypatch.setattr(main_mod, "claude_limits", None)
    monkeypatch.setattr(runner_mod, "DurableRunner", Quiet)
    monkeypatch.setattr(main_mod, "_build_personality", lambda: Drv())
    monkeypatch.setattr(main_mod, "personality", None)

    async def go():
        async with main_mod.lifespan(main_mod.app):
            pass

    asyncio.run(go())
    (patterns,) = patterns_seen
    assert (PERSONALITY_CHOICE_CHANNEL in patterns) is enabled
