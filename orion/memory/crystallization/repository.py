"""Postgres repository for MemoryCrystallizationV1.

Follows the memory_cards storage idiom: idempotent SQL applied at boot
(apply_memory_crystallizations_schema) and a thin synchronous psycopg2 DAL.
The full artifact is persisted losslessly in `doc`; child tables
(claims/sources/links) are queryable projections rebuilt on every upsert.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import psycopg2
import psycopg2.extras

from orion.memory.crystallization.governor import GovernanceHistoryEntry
from orion.schemas.memory_crystallization import (
    ActiveMemoryPacketV1,
    MemoryCrystallizationV1,
)

_SQL_PATH = Path(__file__).parent / "sql" / "memory_crystallizations.sql"


def apply_memory_crystallizations_schema(dsn: str) -> None:
    """Apply idempotent DDL for the crystallization tables."""
    sql = _SQL_PATH.read_text(encoding="utf-8")
    with psycopg2.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
        conn.commit()


class CrystallizationRepository:
    def __init__(self, dsn: str):
        self._dsn = dsn

    def _connect(self):
        return psycopg2.connect(self._dsn)

    # -- canonical writes ------------------------------------------------

    def upsert(self, crystallization: MemoryCrystallizationV1) -> None:
        doc = crystallization.model_dump(mode="json")
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO memory_crystallizations (
                        crystallization_id, kind, subject, summary, status,
                        confidence, salience, scope, tags, grammar_envelope,
                        planning_effects, retrieval_affordances, governance,
                        projection_refs, doc, created_at, updated_at
                    ) VALUES (
                        %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s,
                        %s, %s, %s,
                        %s, %s, %s, %s
                    )
                    ON CONFLICT (crystallization_id) DO UPDATE SET
                        kind = EXCLUDED.kind,
                        subject = EXCLUDED.subject,
                        summary = EXCLUDED.summary,
                        status = EXCLUDED.status,
                        confidence = EXCLUDED.confidence,
                        salience = EXCLUDED.salience,
                        scope = EXCLUDED.scope,
                        tags = EXCLUDED.tags,
                        grammar_envelope = EXCLUDED.grammar_envelope,
                        planning_effects = EXCLUDED.planning_effects,
                        retrieval_affordances = EXCLUDED.retrieval_affordances,
                        governance = EXCLUDED.governance,
                        projection_refs = EXCLUDED.projection_refs,
                        doc = EXCLUDED.doc,
                        updated_at = EXCLUDED.updated_at
                    """,
                    (
                        crystallization.crystallization_id,
                        crystallization.kind,
                        crystallization.subject,
                        crystallization.summary,
                        crystallization.status,
                        crystallization.confidence,
                        crystallization.salience,
                        list(crystallization.scope),
                        list(crystallization.tags),
                        json.dumps(doc["grammar_envelope"]),
                        list(crystallization.planning_effects),
                        list(crystallization.retrieval_affordances),
                        json.dumps(doc["governance"]),
                        json.dumps(doc["projection_refs"]),
                        json.dumps(doc),
                        crystallization.created_at,
                        crystallization.updated_at,
                    ),
                )
                self._rebuild_children(cur, crystallization)
            conn.commit()

    def _rebuild_children(self, cur, crystallization: MemoryCrystallizationV1) -> None:
        cid = crystallization.crystallization_id
        cur.execute("DELETE FROM memory_crystallization_claims WHERE crystallization_id = %s", (cid,))
        for claim in crystallization.claims:
            cur.execute(
                """
                INSERT INTO memory_crystallization_claims (
                    claim_id, crystallization_id, claim, status, confidence, evidence_ref_ids
                ) VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (claim.claim_id, cid, claim.claim, claim.status, claim.confidence, list(claim.evidence_refs)),
            )
        cur.execute("DELETE FROM memory_crystallization_sources WHERE crystallization_id = %s", (cid,))
        for ev in crystallization.evidence:
            cur.execute(
                """
                INSERT INTO memory_crystallization_sources (
                    crystallization_id, source_kind, source_id, excerpt, strength, note
                ) VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (cid, ev.source_kind, ev.source_id, ev.excerpt, ev.strength, ev.note),
            )
        cur.execute("DELETE FROM memory_crystallization_links WHERE from_crystallization_id = %s", (cid,))
        for link in crystallization.links:
            cur.execute(
                """
                INSERT INTO memory_crystallization_links (
                    from_crystallization_id, to_crystallization_id, relation, confidence, note
                ) VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (from_crystallization_id, to_crystallization_id, relation) DO UPDATE SET
                    confidence = EXCLUDED.confidence,
                    note = EXCLUDED.note
                """,
                (cid, link.target_crystallization_id, link.relation, link.confidence, link.note),
            )

    # -- reads -----------------------------------------------------------

    def get(self, crystallization_id: str) -> MemoryCrystallizationV1 | None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT doc FROM memory_crystallizations WHERE crystallization_id = %s",
                    (crystallization_id,),
                )
                row = cur.fetchone()
        if row is None:
            return None
        return MemoryCrystallizationV1.model_validate(row[0])

    def list(
        self,
        *,
        status: str | None = None,
        kind: str | None = None,
        scope: str | None = None,
        limit: int = 100,
    ) -> list[MemoryCrystallizationV1]:
        clauses: list[str] = []
        params: list[Any] = []
        if status:
            clauses.append("status = %s")
            params.append(status)
        if kind:
            clauses.append("kind = %s")
            params.append(kind)
        if scope:
            clauses.append("%s = ANY(scope)")
            params.append(scope)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT doc FROM memory_crystallizations
                    {where}
                    ORDER BY salience DESC, updated_at DESC
                    LIMIT %s
                    """,
                    params,
                )
                rows = cur.fetchall()
        return [MemoryCrystallizationV1.model_validate(row[0]) for row in rows]

    def list_links(self, crystallization_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT from_crystallization_id, to_crystallization_id, relation,
                           confidence, note, created_at
                    FROM memory_crystallization_links
                    WHERE from_crystallization_id = %s OR to_crystallization_id = %s
                    ORDER BY created_at
                    """,
                    (crystallization_id, crystallization_id),
                )
                rows = cur.fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["confidence"] = float(item["confidence"])
            item["created_at"] = item["created_at"].isoformat()
            out.append(item)
        return out

    # -- history / audit ---------------------------------------------------

    def record_history(
        self,
        entry: GovernanceHistoryEntry,
        *,
        before: dict[str, Any] | None = None,
        after: dict[str, Any] | None = None,
    ) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO memory_crystallization_history (
                        crystallization_id, op, actor, before, after, reason
                    ) VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (
                        entry.crystallization_id,
                        entry.op,
                        entry.actor,
                        json.dumps(before) if before is not None else None,
                        json.dumps(after) if after is not None else None,
                        entry.reason,
                    ),
                )
            conn.commit()

    def list_history(self, crystallization_id: str, limit: int = 100) -> list[dict[str, Any]]:
        with self._connect() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT history_id, op, actor, before, after, reason, created_at
                    FROM memory_crystallization_history
                    WHERE crystallization_id = %s
                    ORDER BY created_at DESC
                    LIMIT %s
                    """,
                    (crystallization_id, limit),
                )
                rows = cur.fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["created_at"] = item["created_at"].isoformat()
            out.append(item)
        return out

    # -- retrieval events --------------------------------------------------

    def record_retrieval_event(self, retrieval_event_id: str, packet: ActiveMemoryPacketV1) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO memory_crystallization_retrieval_events (
                        retrieval_event_id, query, packet
                    ) VALUES (%s, %s, %s)
                    ON CONFLICT (retrieval_event_id) DO NOTHING
                    """,
                    (retrieval_event_id, packet.query, json.dumps(packet.model_dump(mode="json"))),
                )
            conn.commit()

    def get_retrieval_event(self, retrieval_event_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT retrieval_event_id, query, packet, created_at
                    FROM memory_crystallization_retrieval_events
                    WHERE retrieval_event_id = %s
                    """,
                    (retrieval_event_id,),
                )
                row = cur.fetchone()
        if row is None:
            return None
        item = dict(row)
        item["created_at"] = item["created_at"].isoformat()
        return item
