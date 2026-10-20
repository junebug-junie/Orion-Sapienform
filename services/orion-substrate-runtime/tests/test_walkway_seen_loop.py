"""Walkway rollups -> "seen on <day>" substrate links for named people (2026-10-18).

Pinned: only existing entity nodes are linked (never created); a copied label links to the
typed individual's node; one day evidence node per (stream, date); re-running adds nothing;
no named person = no record. The SQL is also run against a throwaway Postgres.

The store here is the production FalkorSubstrateStore (on a recording client, or a throwaway
FalkorDB when ORION_TEST_FALKOR_URI is set), not the in-memory store: the in-memory store
accepts any node kind, so the first version of these tests passed while every live pass
raised "durable writes support concept, evidence, entity, assertion nodes only; got
node_kind='event'" (2026-10-18).
"""

from __future__ import annotations

import json
import os
import sys
import uuid
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

# Collected from the repo root by the real-FalkorDB lane (.github/workflows/substrate-neighborhood.yml):
# the service conftest only fixes sys.path per test, after this module's imports already ran.
for _p in (Path(__file__).resolve().parents[3], Path(__file__).resolve().parents[1]):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from orion.core.schemas.cognitive_substrate import EntityNodeV1, EventNodeV1, EvidenceNodeV1
from orion.graph.falkor_client import RecordingFalkorClient
from orion.substrate.adapters.vision_individual import (
    entity_node_id_for_individual,
    map_vision_individual_label_to_substrate,
    map_walkway_day_seen_to_substrate,
    walkway_day_node_id,
)
from orion.substrate.falkor_codec import DURABLE_NODE_KINDS
from orion.substrate.falkor_store import FalkorSubstrateStore, FalkorSubstrateStoreConfig
from orion.substrate.materializer import SubstrateGraphMaterializer
from orion.substrate.reconcile import SubstrateIdentityResolver

import app.walkway_seen_loop as wsl

DAY = date(2026, 10, 16)
START = datetime(2026, 10, 16, 6, 0, tzinfo=timezone.utc)
LINE = "Walkway Fri 16 Oct 2026: 309 passings; named: Robin (13:33, 18:26), Sam (11:56)."


def _falkor_store(cfg: FalkorSubstrateStoreConfig | None = None) -> FalkorSubstrateStore:
    if cfg is not None:
        return FalkorSubstrateStore(cfg, hydrate=False)
    return FalkorSubstrateStore(FalkorSubstrateStoreConfig(uri="redis://localhost:6379", graph_name="t"),
                                client=RecordingFalkorClient(), hydrate=False)


def _store_with(*named: tuple[str, str], store: FalkorSubstrateStore | None = None) -> FalkorSubstrateStore:
    store = store or _falkor_store()
    mat = SubstrateGraphMaterializer(store=store, identity_resolver=SubstrateIdentityResolver(store=store))
    for iid, label in named:
        mat.apply_record(map_vision_individual_label_to_substrate(
            individual_id=iid, label=label, kind="person", ask_id=f"ask-{iid}",
            answered_at=START, stream_id="walkway"))
    return store


class _Result:
    def __init__(self, rows):
        self.rows = rows

    def mappings(self):
        return self

    def all(self):
        return self.rows


class _Conn:
    def __init__(self, days, typed):
        self.days, self.typed = days, typed

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, stmt, params):
        sql = str(stmt)
        if "FROM vision_rollup" in sql:
            return _Result(self.days)
        return _Result([(i,) for i in self.typed.get(params["l"], [])])


class _Engine:
    def __init__(self, days, typed=None):
        self.days, self.typed = days, typed or {}

    def connect(self):
        return _Conn(self.days, self.typed)


def _day_row(named):
    return {"stream_id": "walkway", "period_start": START, "label": DAY.isoformat(), "line": LINE,
            "named": json.dumps(named)}


def test_links_existing_named_entities_to_one_day_node_and_is_idempotent():
    store = _store_with(("ind-robin", "Robin"), ("ind-sam", "Sam"))
    named = {"Robin": {"times": ["13:33", "18:26"], "individual_ids": ["ind-robin"]},
             # a copied-label individual has no entity node; its typed twin does
             "Sam": {"times": ["11:56"], "individual_ids": ["ind-sam-copy"]},
             "Nobody Named Yet": {"times": ["09:00"], "individual_ids": ["ind-unanswered"]}}
    engine = _Engine([_day_row(named)], typed={"Sam": ["ind-sam"]})
    done: dict = {}
    out = wsl.link_recent_days(engine=engine, store=store, streams=("walkway",), lookback_days=3, done=done)
    assert out == [{"day": "2026-10-16", "stream": "walkway", "people": ["Robin", "Sam"], "edges": 2,
                    "edges_created": 2}]
    day_node = store.get_node_by_id(walkway_day_node_id("walkway", DAY))
    assert isinstance(day_node, EvidenceNodeV1) and day_node.metadata["summary"] == LINE
    assert day_node.content_ref == "vision_rollup:walkway:day:2026-10-16"
    # no entity node is created for an individual Juniper never named through an ask
    assert store.get_node_by_id(entity_node_id_for_individual("ind-unanswered")) is None
    assert store.get_node_by_id(entity_node_id_for_individual("ind-sam-copy")) is None
    # same process: the day is skipped; a fresh process (restart) re-applies harmlessly
    assert wsl.link_recent_days(engine=engine, store=store, streams=("walkway",), lookback_days=3, done=done) == []
    again = wsl.link_recent_days(engine=engine, store=store, streams=("walkway",), lookback_days=3, done={})
    assert again[0]["edges_created"] == 0
    # Juniper names the third person later: the same process links them on the next pass.
    SubstrateGraphMaterializer(store=store, identity_resolver=SubstrateIdentityResolver(store=store)).apply_record(
        map_vision_individual_label_to_substrate(individual_id="ind-unanswered", label="Nobody Named Yet",
                                                 kind="person", ask_id="ask-late", answered_at=START,
                                                 stream_id="walkway"))
    later = wsl.link_recent_days(engine=engine, store=store, streams=("walkway",), lookback_days=3, done=done)
    assert later and later[0]["people"] == ["Nobody Named Yet", "Robin", "Sam"] and later[0]["edges_created"] == 1


# A day row as rules_version 1 wrote it (before people became a range, 2026-10-20).
V1_DAY = {
    "schema": "vision_rollup.v1", "rules_version": 1, "stream_id": "walkway", "grain": "day",
    "period_start": "2026-10-16T06:00:00+00:00", "period_end": "2026-10-17T06:00:00+00:00",
    "label": "2026-10-16", "tz": "America/Denver", "complete": True,
    "covered_until": "2026-10-17T06:00:00+00:00",
    "kinds": {"person": {"passings": 137, "distinct": 89, "sightings": 151},
              "vehicle": {"passings": 40, "distinct": 12, "sightings": 60}},
    "total": {"passings": 177, "distinct": 101, "sightings": 211}, "peak_at_once": {"person": 5, "vehicle": 3},
    "camera": {"up_minutes": 1400, "expected_minutes": 1440, "status": "ok", "down_spans": [], "missing_spans": []},
    "busiest": {"at": "12:00", "passings": 92, "peak_at_once": {"person": 5}}, "quietest": None,
    "baseline": {"mean_passings": None, "periods": 0},
}
V1_LINE = ("Walkway Fri 16 Oct 2026: 177 passings (137 people, 40 vehicles) by 101 distinct; "
           "busiest 12:00 (92, most at once 5 people).")
RANGE_TEXT = "people: at least 5, at most 137 (the camera splits one person into several)"


def test_an_old_day_row_lands_in_the_graph_as_a_people_range():
    from orion.vision.walkway_rollup import bare_people_counts

    store = _store_with(("ind-robin", "Robin"))
    row = dict(_day_row({"Robin": {"times": ["13:33"], "individual_ids": ["ind-robin"]}}),
               line=V1_LINE, rollup=json.dumps(V1_DAY))
    assert "rollup FROM vision_rollup" in wsl.DAYS_SQL
    wsl.link_recent_days(engine=_Engine([row]), store=store, streams=("walkway",), lookback_days=3, done={})
    summary = store.get_node_by_id(walkway_day_node_id("walkway", DAY)).metadata["summary"]
    assert RANGE_TEXT in summary and "137 people" not in summary and bare_people_counts(summary) == []


def test_no_named_person_with_an_entity_means_no_record():
    store = _store_with()
    engine = _Engine([_day_row({"Robin": {"individual_ids": ["ind-robin"]}})])
    assert wsl.link_recent_days(engine=engine, store=store, streams=("walkway",), lookback_days=3, done={}) == []
    assert store.get_node_by_id(walkway_day_node_id("walkway", DAY)) is None
    assert map_walkway_day_seen_to_substrate(stream_id="walkway", day=DAY, line=LINE, seen=[],
                                             observed_at=START) is None


def test_edges_are_observed_in_from_entity_to_the_day_with_deterministic_ids():
    rec = map_walkway_day_seen_to_substrate(
        stream_id="walkway", day=DAY, line=LINE, seen=[("entity:vision_individual:abc", "Robin", "13:33")],
        observed_at=START)
    rec2 = map_walkway_day_seen_to_substrate(
        stream_id="walkway", day=DAY, line=LINE, seen=[("entity:vision_individual:abc", "Robin", "13:33")],
        observed_at=START)
    (edge,) = rec.edges
    assert edge.predicate == "observed_in" and edge.source.node_kind == "entity"
    assert edge.target.node_id == "evidence:walkway_day:walkway:2026-10-16" and edge.edge_id == rec2.edges[0].edge_id
    assert edge.target.node_kind == "evidence"
    assert edge.metadata["first_seen_local"] == "13:33"


def test_every_emitted_node_kind_is_one_the_falkor_store_writes_durably():
    rec = map_walkway_day_seen_to_substrate(
        stream_id="walkway", day=DAY, line=LINE, seen=[("entity:vision_individual:abc", "Robin", "13:33")],
        observed_at=START)
    kinds = {n.node_kind for n in rec.nodes} | {r.node_kind for e in rec.edges for r in (e.source, e.target)}
    assert kinds <= set(DURABLE_NODE_KINDS), kinds


def test_regression_an_event_day_node_fails_the_whole_pass_on_the_falkor_store(monkeypatch):
    """Harness guard, not coverage for the fix: the exact live failure (PR #2755 built the day as
    an EventNodeV1 and every pass raised) must still fail in this harness, so these tests can never
    again pass on a store that accepts any node kind. The fix itself is guarded by
    test_every_emitted_node_kind_is_one_the_falkor_store_writes_durably and the main pass test."""
    import orion.substrate.adapters.vision_individual as adapter

    real = adapter.map_walkway_day_seen_to_substrate

    def old_event_day(**kwargs):
        rec = real(**kwargs)
        ev = rec.nodes[0]
        day = EventNodeV1(node_id="event:walkway_day:walkway:2026-10-16", anchor_scope="world",
                          subject_ref=ev.subject_ref, temporal=ev.temporal, provenance=ev.provenance,
                          event_type="walkway_day", summary=ev.metadata["summary"])
        edges = [e.model_copy(update={"target": e.target.model_copy(update={"node_id": day.node_id,
                                                                              "node_kind": "event"})})
                 for e in rec.edges]
        return rec.model_copy(update={"nodes": [day], "edges": edges})

    monkeypatch.setattr(adapter, "map_walkway_day_seen_to_substrate", old_event_day)
    store = _store_with(("ind-robin", "Robin"))
    engine = _Engine([_day_row({"Robin": {"times": ["13:33"], "individual_ids": ["ind-robin"]}})])
    with pytest.raises(ValueError, match="durable writes support concept, evidence, entity, assertion nodes only; "
                                          "got node_kind='event'"):
        wsl.link_recent_days(engine=engine, store=store, streams=("walkway",), lookback_days=3, done={})


FALKOR = os.getenv("ORION_TEST_FALKOR_URI", "")


@pytest.mark.skipif(not FALKOR, reason="ORION_TEST_FALKOR_URI not set (throwaway FalkorDB only)")
def test_real_falkor_writes_observed_in_edges_and_a_fresh_store_hydrates_them():
    import redis

    graph = f"orion_test_walkway_seen_{uuid.uuid4().hex[:10]}"
    cfg = FalkorSubstrateStoreConfig(uri=FALKOR, graph_name=graph)
    try:
        store = _store_with(("ind-robin", "Robin"), store=_falkor_store(cfg))
        engine = _Engine([_day_row({"Robin": {"times": ["13:33"], "individual_ids": ["ind-robin"]}})])
        out = wsl.link_recent_days(engine=engine, store=store, streams=("walkway",), lookback_days=3, done={})
        assert out and out[0]["edges_created"] == 1
        rows = store._client.graph_query(
            "MATCH (p:SubstrateNode)-[e]->(d:SubstrateNode {node_id: $d}) WHERE e.predicate = 'observed_in' "
            "RETURN p.node_id AS person, d.node_kind AS kind, d.content_ref AS ref",
            {"d": walkway_day_node_id("walkway", DAY)})
        assert [(r["person"], r["kind"], r["ref"]) for r in rows] == [
            (entity_node_id_for_individual("ind-robin"), "evidence", "vision_rollup:walkway:day:2026-10-16")]
        fresh = FalkorSubstrateStore(cfg, hydrate=True)
        assert isinstance(fresh.get_node_by_id(walkway_day_node_id("walkway", DAY)), EvidenceNodeV1)
        # a restart (fresh process, empty done) adds no second edge
        again = wsl.link_recent_days(engine=engine, store=fresh, streams=("walkway",), lookback_days=3, done={})
        assert again[0]["edges_created"] == 0
    finally:
        try:
            redis.Redis.from_url(FALKOR).execute_command("GRAPH.DELETE", graph)
        except redis.ResponseError:  # graph never created: keep the real failure visible
            pass


PG = os.getenv("ORION_TEST_PG_URI", "")


@pytest.mark.skipif(not PG, reason="ORION_TEST_PG_URI not set (throwaway Postgres only)")
def test_sql_runs_against_postgres():
    from sqlalchemy import create_engine, text

    schema = "walkway_seen_test"  # own schema: never touch the shared CI database's public tables
    admin = create_engine(PG)
    with admin.begin() as conn:
        conn.execute(text(f"DROP SCHEMA IF EXISTS {schema} CASCADE"))
        conn.execute(text(f"CREATE SCHEMA {schema}"))
    engine = create_engine(f"{PG}{'&' if '?' in PG else '?'}options=-csearch_path%3D{schema}")
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE vision_individual (individual_id TEXT PRIMARY KEY, stream_id TEXT, "
                          "label TEXT, label_copied BOOLEAN NOT NULL DEFAULT false)"))
        conn.execute(text("INSERT INTO vision_individual VALUES ('ind-sam','walkway','Sam',false), "
                          "('ind-sam-copy','walkway','Sam',true)"))
    sql = (Path(__file__).resolve().parents[3] / "services/orion-sql-db/manual_migration_vision_rollup_v1.sql").read_text()
    with engine.begin() as conn:
        conn.execute(text(sql))
        conn.execute(text(
            "INSERT INTO vision_rollup (stream_id, grain, period_start, period_end, label, complete, rollup, line) "
            "VALUES ('walkway','day',:s,:e,'2026-10-16',true,CAST(:j AS jsonb),:l), "
            "('walkway','day',:s2,:e2,'2026-10-17',false,'{}','partial')"),
            {"s": START, "e": START.replace(day=17), "s2": START.replace(day=17), "e2": START.replace(day=18),
             "j": json.dumps({"named": {"Sam": {"times": ["11:56"], "individual_ids": ["ind-sam-copy"]}}}), "l": LINE})
    store = _store_with(("ind-sam", "Sam"))
    out = wsl.link_recent_days(engine=engine, store=store, streams=("walkway",), lookback_days=3, done={},
                               now=datetime(2026, 10, 18, 3, 0, tzinfo=timezone.utc))
    # only the complete day; the copied-label sighting links to the typed individual's node
    assert out == [{"day": "2026-10-16", "stream": "walkway", "people": ["Sam"], "edges": 1, "edges_created": 1}]
    engine.dispose()
    with admin.begin() as conn:
        conn.execute(text(f"DROP SCHEMA IF EXISTS {schema} CASCADE"))
    admin.dispose()
