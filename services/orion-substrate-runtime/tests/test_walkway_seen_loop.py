"""Walkway rollups -> "seen on <day>" substrate links for named people (2026-10-18).

Pinned: only existing entity nodes are linked (never created); a copied label links to the
typed individual's node; one day event node per (stream, date); re-running adds nothing;
no named person = no record. The SQL is also run against a throwaway Postgres.
"""

from __future__ import annotations

import json
import os
from datetime import date, datetime, timezone

import pytest

from orion.core.schemas.cognitive_substrate import EntityNodeV1, EventNodeV1
from orion.substrate.adapters.vision_individual import (
    entity_node_id_for_individual,
    map_vision_individual_label_to_substrate,
    map_walkway_day_seen_to_substrate,
    walkway_day_node_id,
)
from orion.substrate.materializer import SubstrateGraphMaterializer
from orion.substrate.reconcile import SubstrateIdentityResolver
from orion.substrate.store import InMemorySubstrateGraphStore

import app.walkway_seen_loop as wsl

DAY = date(2026, 10, 16)
START = datetime(2026, 10, 16, 6, 0, tzinfo=timezone.utc)
LINE = "Walkway Fri 16 Oct 2026: 309 passings; named: Robin (13:33, 18:26), Sam (11:56)."


def _store_with(*named: tuple[str, str]) -> InMemorySubstrateGraphStore:
    store = InMemorySubstrateGraphStore()
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
    done: set = set()
    out = wsl.link_recent_days(engine=engine, store=store, streams=("walkway",), lookback_days=3, done=done)
    assert out == [{"day": "2026-10-16", "stream": "walkway", "people": ["Robin", "Sam"], "edges": 2,
                    "edges_created": 2}]
    day_node = store.get_node_by_id(walkway_day_node_id("walkway", DAY))
    assert isinstance(day_node, EventNodeV1) and day_node.summary == LINE
    # no entity node is created for an individual Juniper never named through an ask
    assert store.get_node_by_id(entity_node_id_for_individual("ind-unanswered")) is None
    assert store.get_node_by_id(entity_node_id_for_individual("ind-sam-copy")) is None
    # same process: the day is skipped; a fresh process (restart) re-applies harmlessly
    assert wsl.link_recent_days(engine=engine, store=store, streams=("walkway",), lookback_days=3, done=done) == []
    again = wsl.link_recent_days(engine=engine, store=store, streams=("walkway",), lookback_days=3, done=set())
    assert again[0]["edges_created"] == 0


def test_no_named_person_with_an_entity_means_no_record():
    store = _store_with()
    engine = _Engine([_day_row({"Robin": {"individual_ids": ["ind-robin"]}})])
    assert wsl.link_recent_days(engine=engine, store=store, streams=("walkway",), lookback_days=3, done=set()) == []
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
    assert edge.target.node_id == "event:walkway_day:walkway:2026-10-16" and edge.edge_id == rec2.edges[0].edge_id
    assert edge.metadata["first_seen_local"] == "13:33"


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
    from pathlib import Path

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
    out = wsl.link_recent_days(engine=engine, store=store, streams=("walkway",), lookback_days=3, done=set(),
                               now=datetime(2026, 10, 18, 3, 0, tzinfo=timezone.utc))
    # only the complete day; the copied-label sighting links to the typed individual's node
    assert out == [{"day": "2026-10-16", "stream": "walkway", "people": ["Sam"], "edges": 1, "edges_created": 1}]
    engine.dispose()
    with admin.begin() as conn:
        conn.execute(text(f"DROP SCHEMA IF EXISTS {schema} CASCADE"))
    admin.dispose()
