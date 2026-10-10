"""Project memory referents from Postgres into the substrate graph (memory Stage 2 spec 2.3).

The single writer of ``memory.referents`` nodes and their provenance edges. Postgres is
the truth (``referent_alias``, ``episode_memory*``, ``substrate_graph_journal``); every
write goes through ``SubstrateGraphMaterializer.apply_record`` and is idempotent, so the
graph can be rebuilt by truncating ``referent_projection``.

One tick:
0. readiness: nothing is written until every substrate reader advertises the
   assertion-core capability (orion/substrate/reader_capability.py). While waiting it logs
   ``referent_projector_waiting reason=readers_not_ready missing=[...]`` and /health shows it;
1. nodes: one Entity (or Concept, for ``concept:`` keys) per referent node; state = its
   key row's state; display aliases = its live names. On a node's FIRST projection, if
   another producer already has a same-named Concept/Entity (topic-foundry's "circe"),
   Orion gets an identity question; the node stays walkable. Never a merge, never a demotion;
2. memories: one Evidence node per memory (``episode_memory:<id>``; the text stays in
   Postgres) and an ``observed_in`` provenance edge from each of its referent nodes. A
   memory that stops being active closes its edges (``valid_to``) instead of deleting;
3. assertions: ``AssertionProjector.run_once`` applies co-occurrence decisions.

Kill switch: MEMORY_REFERENT_PROJECTOR_ENABLED=false. Rebuild the graph from Postgres:
``ReferentProjector.rebuild()`` (CLI: scripts/rebuild_referent_graph.py), never by hand:
truncating the ledger alone does not re-project accepted assertions.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from orion.core.schemas.cognitive_substrate import (
    ConceptNodeV1,
    EntityNodeV1,
    EvidenceNodeV1,
    NodeRefV1,
    SubstrateEdgeV1,
    SubstrateGraphRecordV1,
    SubstrateProvenanceV1,
    SubstrateTemporalWindowV1,
)
from orion.memory.referents.aliases import slug_text
from orion.memory.referents.cooccurrence import evidence_node_id
from orion.memory.referents.resolve import (
    JUNIPER_ANSWERS_KINDS,
    LIVE_STATES,
    REFERENT_NAMESPACE,
    REFERENT_PRODUCER,
)
from orion.core.schemas.substrate_graph_journal import SubstrateGraphDecisionV1
from orion.substrate.assertion_projector import AssertionProjector
from orion.substrate.graph_journal import SubstrateGraphJournal
from orion.substrate.materializer import SubstrateGraphMaterializer
from orion.substrate.reader_capability import ReadinessCheck, ReadinessV1

logger = logging.getLogger(__name__)

_ALIASES = """
SELECT node_id, alias_norm, alias_text, alias_class, referent_kind, promotion_state, grounded_in, valid_until,
       created_at, updated_at
FROM referent_alias ORDER BY node_id, created_at, alias_norm
"""
_LEDGER = "SELECT subject_id, fingerprint FROM referent_projection WHERE subject_kind = $1"
_SAVE_LEDGER = """
INSERT INTO referent_projection (subject_id, subject_kind, fingerprint, projected_at) VALUES ($1, $2, $3, $4)
ON CONFLICT (subject_id) DO UPDATE SET fingerprint = EXCLUDED.fingerprint, projected_at = EXCLUDED.projected_at
"""
_QUESTION = """
INSERT INTO memory_tension_shadow (question_id, text, kind, scope, answer_via, source_refs, referent_keys, created_at)
VALUES ($1::uuid, $2, 'question', $3, $4, $5::jsonb, $6, $7) ON CONFLICT (question_id) DO NOTHING
"""
# The decision behind each target's latest APPLIED revision (for rebuild()).
_LATEST_APPLIED = """
SELECT DISTINCT ON (d.target_id) d.payload
FROM substrate_graph_journal d
JOIN substrate_graph_journal m ON m.event_kind = 'materialization' AND m.outcome = 'applied'
                              AND m.decision_id = d.decision_id
WHERE d.event_kind = 'decision' AND d.actor = 'memory.referents'
ORDER BY d.target_id, d.revision DESC
"""
_MEMORIES = """
SELECT m.memory_id::text AS memory_id, m.status, m.voice, m.created_at, m.updated_at,
       array_agg(DISTINCT r.node_id ORDER BY r.node_id) AS node_ids
FROM episode_memory m JOIN episode_memory_referent r USING (memory_id)
WHERE r.node_id IS NOT NULL
GROUP BY m.memory_id, m.status, m.voice, m.created_at, m.updated_at
"""


def _fp(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def _edge_id(*parts: str) -> str:
    return f"edge-{uuid.uuid5(REFERENT_NAMESPACE, '|'.join(parts))}"


@dataclass
class ProjectionTickV1:
    blocked: ReadinessV1 | None = None    # readers not ready: nothing was written this tick
    nodes: int = 0
    label_questions: list[str] = field(default_factory=list)
    memories: int = 0
    waiting_memories: int = 0
    assertions_applied: int = 0
    assertions_failed: int = 0


class ReferentProjector:
    def __init__(self, *, pool: Any, materializer: SubstrateGraphMaterializer, readiness: ReadinessCheck) -> None:
        self._pool = pool
        self._materializer = materializer
        self._store = materializer.store
        self._readiness = readiness
        self.last_readiness: ReadinessV1 | None = None
        # Only memory's own claims: this store primes memory nodes only, so it cannot
        # see a reading claim's endpoints (the Hub applies those; see
        # orion/world_pulse_read/assertions.py).
        self._assertions = AssertionProjector(journal=SubstrateGraphJournal(pool), materializer=materializer,
                                              readiness=readiness, proposal_actors=(REFERENT_PRODUCER,))

    async def run_once(self, *, now: datetime | None = None) -> ProjectionTickV1:
        now = now or datetime.now(timezone.utc)
        tick = ProjectionTickV1()
        ready = await asyncio.to_thread(self._readiness)
        self.last_readiness = ready
        if not ready.ready:
            tick.blocked = ready
            logger.info("referent_projector_waiting reason=%s missing=%s", ready.reason, list(ready.missing))
            return tick
        projected = await self._project_nodes(now, tick)
        await self._project_memories(now, projected, tick)
        report = await self._assertions.run_once()
        tick.assertions_applied, tick.assertions_failed = len(report.applied), len(report.failed)
        logger.info(
            "referent_projection_tick nodes=%d label_questions=%d memories=%d waiting_memories=%d "
            "assertions_applied=%d assertions_failed=%d",
            tick.nodes, len(tick.label_questions), tick.memories, tick.waiting_memories,
            tick.assertions_applied, tick.assertions_failed,
        )
        return tick

    # ── nodes ───────────────────────────────────────────────────────────────

    async def _project_nodes(self, now: datetime, tick: ProjectionTickV1) -> dict[str, str]:
        """Returns node_id -> node kind ("entity" | "concept") for every node in the graph."""
        async with self._pool.acquire() as conn:
            rows = [dict(r) for r in await conn.fetch(_ALIASES)]
            ledger = {r["subject_id"]: r["fingerprint"] for r in await conn.fetch(_LEDGER, "node")}
        by_node: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            by_node[row["node_id"]].append(row)
        projected: dict[str, str] = {}
        for node_id, node_rows in sorted(by_node.items()):
            keys = [r for r in node_rows if r["alias_class"] == "key"]
            if not keys:
                continue
            key = keys[0]
            node_kind = "concept" if key["referent_kind"] == "concept" else "entity"
            if node_id in ledger:
                projected[node_id] = node_kind
            if node_id not in ledger and await self._ask_on_label_collision(node_id, key, now):
                tick.label_questions.append(node_id)
            names = sorted({r["alias_text"] for r in node_rows if r["alias_class"] != "key"
                            and r["promotion_state"] in LIVE_STATES
                            and (r["alias_class"] != "descriptor" or (r["valid_until"] and r["valid_until"] > now))})
            spec = {"kind": key["referent_kind"], "label": slug_text(key["alias_norm"]),
                    "state": key["promotion_state"], "aliases": names}
            fingerprint = _fp(spec)
            if ledger.get(node_id) == fingerprint:
                continue
            grounded = any(r["grounded_in"] for r in node_rows)
            first_seen = min(r["created_at"] for r in node_rows)
            node = self._node(node_id, spec, grounded=grounded, first_seen=first_seen, now=now)
            await asyncio.to_thread(self._materializer.apply_record,
                                    SubstrateGraphRecordV1(anchor_scope="juniper", nodes=[node]))
            async with self._pool.acquire() as conn:
                await conn.execute(_SAVE_LEDGER, node_id, "node", fingerprint, now)
            projected[node_id] = node_kind
            tick.nodes += 1
        return projected

    @staticmethod
    def _node(node_id: str, spec: dict, *, grounded: bool, first_seen: datetime, now: datetime):
        common = dict(
            node_id=node_id, label=spec["label"], anchor_scope="juniper", promotion_state=spec["state"],
            temporal=SubstrateTemporalWindowV1(observed_at=now, valid_from=first_seen),
            provenance=SubstrateProvenanceV1(
                authority="user_asserted" if grounded else "local_inferred",
                source_kind="episode_memory_referent", source_channel="referent_alias", producer=REFERENT_PRODUCER),
        )
        if spec["kind"] == "concept":
            return ConceptNodeV1(**common)
        return EntityNodeV1(entity_type=spec["kind"], aliases=spec["aliases"], **common)

    async def _ask_on_label_collision(self, node_id: str, key: dict, now: datetime) -> bool:
        """Another producer has a same-named node: ask whether they are the same. Our node
        keeps its state (it stays walkable); the rule that asked is recorded on the question.
        Idempotent (deterministic question id), so a rebuild re-asks nothing new."""
        label = slug_text(key["alias_norm"])
        others = await asyncio.to_thread(self._store.find_semantic_node_ids_by_label, label,
                                         exclude_producers=(REFERENT_PRODUCER,))
        if not others:
            return False
        juniper = key["referent_kind"] in JUNIPER_ANSWERS_KINDS
        text = f"Is the '{label}' from our conversations the same as the '{label}' already in my graph?"
        question_id = str(uuid.uuid5(REFERENT_NAMESPACE, f"label_collision|{node_id}|{'|'.join(others)}"))
        async with self._pool.acquire() as conn:
            await conn.execute(
                _QUESTION, question_id, text, "juniper" if juniper else "self",
                "conversation" if juniper else "investigation",
                json.dumps([{"reason": "label_collision", "admitted_by": "label_collision",
                             "node_ids": [node_id, *others]}]),
                [key["alias_norm"]], now)
        logger.info("referent_label_collision_question node_id=%s label=%s others=%s", node_id, label, others)
        return True

    async def rebuild(self, *, now: datetime | None = None) -> ProjectionTickV1:
        """Re-project everything from Postgres into the (possibly empty) graph.

        Truncating ``referent_projection`` re-projects nodes and memory evidence, but NOT
        assertions: the journal (append-only) already records them as applied, so the
        assertion projector has nothing pending. Here each target's latest applied decision
        is replayed through the same record builder, with no new journal rows.
        """
        now = now or datetime.now(timezone.utc)
        async with self._pool.acquire() as conn:
            await conn.execute("TRUNCATE referent_projection")
        tick = await self.run_once(now=now)
        if tick.blocked is not None:
            return tick
        journal = self._assertions._journal
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(_LATEST_APPLIED)
        for row in rows:
            decision = SubstrateGraphDecisionV1.model_validate(json.loads(row["payload"]) if isinstance(
                row["payload"], str) else row["payload"])
            proposal = await journal.proposal(decision.proposal_id)
            if proposal is None:
                continue
            record, _n, _e = await asyncio.to_thread(self._assertions._record, proposal, decision)
            await asyncio.to_thread(self._materializer.apply_record, record)
            tick.assertions_applied += 1
        logger.info("referent_projection_rebuilt assertions_replayed=%d", tick.assertions_applied)
        return tick

    # ── memories ────────────────────────────────────────────────────────────

    async def _project_memories(self, now: datetime, projected: dict[str, str], tick: ProjectionTickV1) -> None:
        async with self._pool.acquire() as conn:
            rows = [dict(r) for r in await conn.fetch(_MEMORIES)]
            ledger = {r["subject_id"]: r["fingerprint"] for r in await conn.fetch(_LEDGER, "memory")}
        for row in rows:
            node_ids = list(row["node_ids"] or [])
            fingerprint = _fp({"status": row["status"], "nodes": node_ids})
            if ledger.get(row["memory_id"]) == fingerprint:
                continue
            if not set(node_ids) <= projected.keys():
                # An endpoint is not in the graph yet; writing the edge would MERGE a
                # placeholder node. Wait for the node pass.
                tick.waiting_memories += 1
                continue
            await asyncio.to_thread(self._materializer.apply_record,
                                    self._memory_record(row, {n: projected[n] for n in node_ids}, now))
            async with self._pool.acquire() as conn:
                await conn.execute(_SAVE_LEDGER, row["memory_id"], "memory", fingerprint, now)
            tick.memories += 1

    @staticmethod
    def _memory_record(row: dict, node_kinds: dict[str, str], now: datetime) -> SubstrateGraphRecordV1:
        content_ref = f"episode_memory:{row['memory_id']}"
        learned = row["created_at"]
        closed = None if row["status"] == "active" else max(row["updated_at"], learned)
        provenance = SubstrateProvenanceV1(
            authority="local_inferred", source_kind="episode_memory", source_channel=str(row["voice"]),
            producer=REFERENT_PRODUCER, evidence_refs=[content_ref])
        evidence = EvidenceNodeV1(
            node_id=evidence_node_id(row["memory_id"]), evidence_type="episode_memory", content_ref=content_ref,
            anchor_scope="juniper", temporal=SubstrateTemporalWindowV1(observed_at=now, valid_from=learned,
                                                                      valid_to=closed),
            provenance=provenance)
        ev_ref = NodeRefV1(node_id=evidence.node_id, node_kind="evidence")
        edges = []
        for node_id, kind in sorted(node_kinds.items()):
            edges.append(SubstrateEdgeV1(
                edge_id=_edge_id(node_id, content_ref, "observed_in"),
                source=NodeRefV1(node_id=node_id, node_kind=kind), target=ev_ref, predicate="observed_in",
                edge_role="provenance",
                temporal=SubstrateTemporalWindowV1(observed_at=now, valid_from=learned, valid_to=closed),
                provenance=provenance))
        return SubstrateGraphRecordV1(anchor_scope="juniper", nodes=[evidence], edges=edges)


def build_projector(pool: Any, settings: Any) -> ReferentProjector | None:
    """The projector on its own Falkor store (no full hydration: it primes only its own nodes),
    gated on every substrate reader advertising the assertion-core capability."""
    uri = str(getattr(settings, "FALKORDB_URI", "") or "").strip()
    if not uri:
        logger.warning("referent_projector_disabled reason=no_FALKORDB_URI")
        return None
    from orion.substrate.falkor_store import FalkorSubstrateStore, FalkorSubstrateStoreConfig
    from orion.substrate.reader_capability import readiness, required_readers_from_env

    store = FalkorSubstrateStore(
        FalkorSubstrateStoreConfig(uri=uri, graph_name=settings.FALKORDB_SUBSTRATE_GRAPH,
                                   client_socket_timeout_s=10.0, client_socket_connect_timeout_s=5.0),
        hydrate=False,
    )
    nodes, edges = store.prime_cache_for_producers((REFERENT_PRODUCER,))
    logger.info("referent_projector_primed graph=%s nodes=%d edges=%d", settings.FALKORDB_SUBSTRATE_GRAPH,
                nodes, edges)
    required = required_readers_from_env()
    return ReferentProjector(pool=pool, materializer=SubstrateGraphMaterializer(store=store),
                             readiness=lambda: readiness(uri, required))


# Read by app.main's /health: {"state": ..., "readiness": {...}} -- "waiting" names the readers.
PROJECTOR_STATUS: dict[str, Any] = {"state": "not_started"}


def record_status(tick: ProjectionTickV1, readiness: ReadinessV1 | None) -> None:
    PROJECTOR_STATUS.update(state="waiting" if tick.blocked else "running",
                            readiness=readiness.as_dict() if readiness else None)


async def run_referent_projector_loop(pool: Any, settings: Any) -> None:
    projector = None
    while projector is None:
        try:
            projector = await asyncio.to_thread(build_projector, pool, settings)
        except Exception:  # noqa: BLE001 - Falkor down at boot: retry, never crash the service
            logger.exception("referent_projector_start_failed")
            PROJECTOR_STATUS.update(state="start_failed")
        if projector is None:
            if not str(getattr(settings, "FALKORDB_URI", "") or "").strip():
                PROJECTOR_STATUS.update(state="disabled_no_falkor_uri")
                return
            await asyncio.sleep(float(settings.MEMORY_REFERENT_PROJECTOR_TICK_SEC))
    while True:
        try:
            tick = await projector.run_once()
            record_status(tick, projector.last_readiness)
        except Exception:  # noqa: BLE001 - one bad tick must not stop projection
            logger.exception("referent_projection_tick_failed")
            PROJECTOR_STATUS.update(state="tick_failed")
        await asyncio.sleep(float(settings.MEMORY_REFERENT_PROJECTOR_TICK_SEC))
