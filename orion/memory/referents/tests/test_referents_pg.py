"""Referents end to end on a disposable Postgres + FalkorDB.

ORION_MEMORY_EPISODE_TEST_DATABASE_URL: admin DSN of a THROWAWAY Postgres (each test makes
and drops its own database). ORION_TEST_FALKOR_URI: a THROWAWAY FalkorDB (each test uses its
own graph). CI: .github/workflows/orion-memory-episode-tests.yml, which fails on any skip.

The fixture mirrors the live 2026-10-06 shapes: Hecate's grounded hardware names, "my boss",
a topic-foundry entity already named "circe", people named together at an offsite.
"""

from __future__ import annotations

import asyncio
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

from orion.memory.episode.store import persist_episode
from orion.memory.episode.validate import EpisodeTurn, validate_distillation
from orion.memory.referents.resolve import ReferentPolicy, node_id_for_key
from orion.schemas.memory_episode import EpisodeDistillationV1

ADMIN_DSN = os.environ.get("ORION_MEMORY_EPISODE_TEST_DATABASE_URL")
FALKOR_URI = os.environ.get("ORION_TEST_FALKOR_URI", "").strip()
SQL = Path(__file__).resolve().parents[4] / "services" / "orion-sql-db"
pytestmark = pytest.mark.skipif(not (ADMIN_DSN and FALKOR_URI),
                                reason="ORION_MEMORY_EPISODE_TEST_DATABASE_URL / ORION_TEST_FALKOR_URI not set")

NOW = datetime(2026, 10, 6, 12, tzinfo=timezone.utc)
HECATE_PROMPT = ("I got us an Inspur NF5288M5 AGX-2 GPU that holds 8x smx2 gpus. We'll call it Hecate "
                 "and it sits next to Circe")
OFFSITE_PROMPT = "my boss Rachel and Vincent are both going to the offsite in Austin"
TURNS = [
    EpisodeTurn(label="t1", correlation_id="c-hecate", prompt=HECATE_PROMPT, response="That is a big upgrade."),
    EpisodeTurn(label="t2", correlation_id="c-offsite", prompt=OFFSITE_PROMPT, response="Have fun at the offsite."),
]
DISTILLATION = {
    "memories": [
        {"purpose": "happened", "voice": "juniper_said", "channel": "chat",
         "statement": "Juniper got an Inspur NF5288M5 server for us and named it Hecate.",
         "stakes": "low", "stakes_reason": "none",
         "referents": [{"key": "project:hecate", "role": "about",
                        "aliases": ["Inspur NF5288M5", "AGX-2 GPU", "8x smx2 gpus", "the new server"]},
                       {"key": "project:circe", "role": "about", "aliases": []}],
         "evidence": [{"turn": "t1", "field": "prompt", "quote": "We'll call it Hecate and it sits next to Circe"},
                      {"turn": "t1", "field": "prompt",
                       "quote": "Inspur NF5288M5 AGX-2 GPU that holds 8x smx2 gpus"}]},
        {"purpose": "happened", "voice": "juniper_said", "channel": "chat",
         "statement": "Juniper's boss Rachel and Vincent are going to the offsite in Austin.",
         "stakes": "low", "stakes_reason": "none",
         "referents": [{"key": "person:rachel", "role": "participant", "aliases": ["my boss"]},
                       {"key": "person:vincent", "role": "participant", "aliases": []},
                       {"key": "event:austin-offsite", "role": "event", "aliases": ["offsite"]},
                       {"key": "person:juniper", "role": "subject", "aliases": []}],
         "evidence": [{"turn": "t2", "field": "prompt",
                       "quote": "my boss Rachel and Vincent are both going to the offsite"}]},
    ],
    "questions": [],
}


def _statements(path: Path) -> list[str]:
    code = "\n".join(re.sub(r"--.*$", "", line) for line in path.read_text().splitlines())
    return [s for s in code.split(";") if s.strip()]


CHAT_LOG = """
CREATE TABLE chat_history_log (id TEXT PRIMARY KEY, correlation_id TEXT, source TEXT, prompt TEXT, response TEXT,
                               created_at TIMESTAMPTZ DEFAULT now())
"""


async def _with_dbs(fn):
    import asyncpg
    import psycopg
    from psycopg.rows import dict_row
    from psycopg_pool import AsyncConnectionPool

    name = f"refs_{uuid.uuid4().hex[:10]}"
    async with await psycopg.AsyncConnection.connect(ADMIN_DSN, autocommit=True) as admin:
        await admin.execute(f'CREATE DATABASE "{name}"')
    dsn = ADMIN_DSN.rsplit("/", 1)[0] + f"/{name}"
    pg = AsyncConnectionPool(conninfo=dsn, min_size=1, max_size=2, open=False,
                             kwargs={"autocommit": True, "row_factory": dict_row})
    await pg.open()
    apg = await asyncpg.create_pool(dsn=dsn, min_size=1, max_size=2)
    try:
        async with pg.connection() as conn:
            for f in ("manual_migration_episode_memory_v1.sql", "manual_migration_substrate_graph_journal_v1.sql",
                      "manual_migration_referent_alias_v1.sql"):
                for stmt in _statements(SQL / f):
                    await conn.execute(stmt)
            await conn.execute(CHAT_LOG)
            # Juniper's prompt corpus: "my"/"the" are common, the proper names are rare.
            for i in range(300):
                await conn.execute("INSERT INTO chat_history_log (id, prompt) VALUES (%s, %s)",
                                   (f"filler-{i}", "my day was fine and the weather was nice"))
            for turn in TURNS:
                await conn.execute("INSERT INTO chat_history_log (id, correlation_id, prompt) VALUES (%s, %s, %s)",
                                   (turn.correlation_id, turn.correlation_id, turn.prompt))
        await fn(pg, apg)
    finally:
        await apg.close()
        await pg.close()
        async with await psycopg.AsyncConnection.connect(ADMIN_DSN, autocommit=True) as admin:
            await admin.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')


def _falkor():
    from orion.graph.falkor_client import RedisGraphQueryClient
    from orion.substrate.falkor_store import FalkorSubstrateStore, FalkorSubstrateStoreConfig

    graph = f"t_refs_{uuid.uuid4().hex[:10]}"
    client = RedisGraphQueryClient(uri=FALKOR_URI, graph_name=graph)
    return graph, client, FalkorSubstrateStore(FalkorSubstrateStoreConfig(uri=FALKOR_URI, graph_name=graph),
                                               client=client, hydrate=False)


def _seed_topic_foundry_circe(store) -> None:
    from orion.core.schemas.cognitive_substrate import (
        EntityNodeV1, SubstrateProvenanceV1, SubstrateTemporalWindowV1)

    store.upsert_node(identity_key="entity|world||label:circe", node=EntityNodeV1(
        node_id="tf-circe", label="circe", anchor_scope="world",
        temporal=SubstrateTemporalWindowV1(observed_at=NOW),
        provenance=SubstrateProvenanceV1(authority="local_inferred", source_kind="topic_foundry.run_topic",
                                         source_channel="t", producer="topic_foundry_adapter")))


async def _persist(pg, policy=ReferentPolicy()):
    result = validate_distillation(EpisodeDistillationV1.model_validate(DISTILLATION), TURNS, episode_id="ep-1")
    assert len(result.memories) == 2 and not result.rejections
    return await persist_episode(pg, episode_id="ep-1", run_id="memdistill-ep-1", result=result,
                                 model_route="memory_distill", model="t", prompt_version="memory_episode_distill.v3",
                                 usage={}, llm_latency_ms=1, hold_wait_ms=0, coverage=1.0, now=NOW,
                                 referent_policy=policy)


async def _aliases(pg) -> dict[tuple[str, str], tuple]:
    async with pg.connection() as conn:
        rows = await (await conn.execute(
            "SELECT node_id, alias_norm, alias_class, promotion_state, admitted_by, valid_until FROM referent_alias"
        )).fetchall()
    return {(r["node_id"], r["alias_norm"]): (r["alias_class"], r["promotion_state"], r["admitted_by"],
                                              r["valid_until"]) for r in rows}


async def _table(pg, sql) -> list[dict]:
    async with pg.connection() as conn:
        return [dict(r) for r in await (await conn.execute(sql)).fetchall()]


HECATE, CIRCE = node_id_for_key("project:hecate"), node_id_for_key("project:circe")
RACHEL, VINCENT = node_id_for_key("person:rachel"), node_id_for_key("person:vincent")
OFFSITE = node_id_for_key("event:austin-offsite")


def test_persist_resolves_every_referent_and_keeps_juniper_s_names():
    async def body(pg, apg):
        counts = await _persist(pg)
        assert counts["referents_resolved"] == 6
        aliases = await _aliases(pg)
        for name in ("hecate", "inspur nf5288m5", "agx-2 gpu", "8x smx2 gpus"):
            assert aliases[(HECATE, name)][:3] == ("name", "provisional", "alias_grounding_v1"), name
        assert aliases[(HECATE, "the new server")][1:3] == ("proposed", "ungrounded")
        boss = aliases[(RACHEL, "my boss")]
        assert boss[:3] == ("descriptor", "provisional", "alias_grounding_v1")
        assert (boss[3] - NOW).days == 90
        unresolved = await _table(pg, "SELECT * FROM episode_memory_referent WHERE node_id IS NULL")
        assert unresolved == []
        decisions = await _table(pg, "SELECT p.payload->>'statement_key' AS k FROM substrate_graph_journal p "
                                     "JOIN substrate_graph_journal d ON d.proposal_id = p.proposal_id "
                                     "AND d.event_kind = 'decision' WHERE p.event_kind = 'proposal'")
        keys = {d["k"] for d in decisions}
        # Hecate-Circe (one quote), and the three offsite pairs; never with Juniper.
        assert len(keys) == 4 and not any(node_id_for_key("person:juniper") in k for k in keys)
        # Replaying the persist writes nothing new.
        before = (await _aliases(pg), await _table(pg, "SELECT event_id FROM substrate_graph_journal ORDER BY 1"))
        await _persist(pg)
        assert (await _aliases(pg), await _table(pg, "SELECT event_id FROM substrate_graph_journal ORDER BY 1")) == before
    asyncio.run(_with_dbs(body))


def test_the_kill_switches_flip_aliases_and_claims_to_proposed():
    async def body(pg, apg):
        await _persist(pg, ReferentPolicy(grounding_auto_accept=False, cooccurrence_auto_accept=False))
        aliases = await _aliases(pg)
        assert aliases[(HECATE, "inspur nf5288m5")][1:3] == ("proposed", "alias_grounding_v1_disabled")
        # No live names, so no claims can name both sides; and no decisions either way.
        assert await _table(pg, "SELECT 1 FROM substrate_graph_journal WHERE event_kind = 'decision'") == []
    asyncio.run(_with_dbs(body))


def test_a_referent_fault_never_loses_the_memories():
    async def body(pg, apg):
        async with pg.connection() as conn:
            await conn.execute("DROP TABLE referent_alias")
        counts = await _persist(pg)
        assert counts["memories"] == 2 and "referents_resolved" not in counts
        assert len(await _table(pg, "SELECT memory_id FROM episode_memory")) == 2
    asyncio.run(_with_dbs(body))


def test_checkpoint_backfill_recovers_aliases_and_is_idempotent():
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

    from orion.memory.referents.backfill import backfill_all

    async def body(pg, apg):
        # Stage 1 state: memories stored, aliases dropped, no node ids.
        await _persist(pg, policy=None)
        assert await _table(pg, "SELECT 1 FROM referent_alias") == []
        import json
        type_, blob = JsonPlusSerializer().dumps_typed("```json\n" + json.dumps(DISTILLATION) + "\n```")
        async with pg.connection() as conn:
            await conn.execute("CREATE TABLE checkpoint_writes (thread_id TEXT, checkpoint_ns TEXT, checkpoint_id TEXT,"
                               " task_id TEXT, idx INT, channel TEXT, type TEXT, blob BYTEA, task_path TEXT)")
            await conn.execute("INSERT INTO checkpoint_writes VALUES ('memdistill-ep-1', '', 'c1', 't', 0, "
                               "'answer_text', %s, %s, '')", (type_, blob))

        async def state():
            return [await _table(pg, sql) for sql in (
                "SELECT * FROM referent_alias ORDER BY node_id, alias_norm",
                "SELECT memory_id, referent_key, role, node_id FROM episode_memory_referent ORDER BY 1, 2, 3",
                "SELECT event_id, payload FROM substrate_graph_journal ORDER BY event_id",
                "SELECT question_id, text FROM memory_tension_shadow ORDER BY question_id")]

        async with pg.connection() as conn:
            first = await backfill_all(conn, now=NOW, policy=ReferentPolicy())
        assert first[0]["answers_recovered"] == 1 and first[0]["resolved"] == 6
        after_one = await state()
        aliases = await _aliases(pg)
        assert aliases[(HECATE, "inspur nf5288m5")][1:3] == ("provisional", "alias_grounding_v1")
        async with pg.connection() as conn:
            await backfill_all(conn, now=NOW, policy=ReferentPolicy())
        assert await state() == after_one
    asyncio.run(_with_dbs(body))


def test_the_daily_report_shows_held_claims_names_by_rule_and_open_questions():
    import copy
    from datetime import timedelta

    from orion.memory.episode.report import render_referents

    variant = copy.deepcopy(DISTILLATION)
    # Juniper says "my boss" about Rachel; the writer also files a second person under it.
    variant["memories"][1]["referents"].append({"key": "person:dana", "role": "about", "aliases": ["my boss"]})

    async def body(pg, apg):
        result = validate_distillation(EpisodeDistillationV1.model_validate(variant), TURNS, episode_id="ep-1")
        await persist_episode(pg, episode_id="ep-1", run_id="r", result=result, model_route="m", model="t",
                              prompt_version="memory_episode_distill.v3", usage={}, llm_latency_ms=1, hold_wait_ms=0,
                              coverage=1.0, now=NOW,
                              referent_policy=ReferentPolicy(cooccurrence_auto_accept=False))
        lines = "\n".join(await render_referents(apg, start=NOW - timedelta(hours=1), end=NOW + timedelta(hours=1)))
        assert "provisional (alias_grounding_v1)" in lines
        assert "Co-occurrence claims held for review (not walkable): " in lines
        assert "held for review (not walkable): 0" not in lines  # the kill switch held them all
        assert "[alias_collision, ask via conversation] Does 'my boss' mean 'rachel' or 'dana'?" in lines
    asyncio.run(_with_dbs(body))
