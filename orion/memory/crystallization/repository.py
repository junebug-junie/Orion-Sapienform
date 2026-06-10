"""Postgres repository for MemoryCrystallizationV1.

Follows the memory_cards storage idiom: idempotent SQL applied at boot
(apply_memory_crystallizations_schema) and a thin synchronous psycopg2 DAL
(callers in async contexts wrap these in asyncio.to_thread).

The full artifact is persisted losslessly in `doc`; child tables
(claims/sources/links) are queryable projections rebuilt on every upsert.

Governance writes are transactional: a status transition and its audit
history row commit together (apply_transition / apply_supersession), and
optimistic status guards prevent stale writers from reverting governed
state (e.g. a bus proposal re-ingest overwriting an approval).
"""

from __future__ import annotations

import json
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Any, Iterable

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
    with closing(psycopg2.connect(dsn)) as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
        conn.commit()


class CrystallizationRepository:
    def __init__(self, dsn: str):
        self._dsn = dsn

    @contextmanager
    def _conn(self):
        with closing(psycopg2.connect(self._dsn)) as conn:
            yield conn

    # -- canonical writes ------------------------------------------------

    def upsert(
        self,
        crystallization: MemoryCrystallizationV1,
        *,
        expected_statuses: Iterable[str] | None = None,
    ) -> bool:
        """Insert or update a crystallization.

        When expected_statuses is given, an existing row is only updated if
        its current status is one of them (optimistic guard). Returns True
        if a row was written.
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                written = self._upsert_on_cursor(
                    cur, crystallization, expected_statuses=expected_statuses
                )
            if written:
                conn.commit()
            else:
                conn.rollback()
        return written

    def apply_transition(
        self,
        crystallization: MemoryCrystallizationV1,
        entry: GovernanceHistoryEntry,
        *,
        before: dict[str, Any] | None = None,
        after: dict[str, Any] | None = None,
        expected_statuses: Iterable[str] | None = None,
    ) -> bool:
        """Write a governance transition and its audit row atomically.

        Returns False (and writes nothing) if the status guard fails.
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                written = self._upsert_on_cursor(
                    cur, crystallization, expected_statuses=expected_statuses
                )
                if written:
                    self._history_on_cursor(cur, entry, before=before, after=after)
            if written:
                conn.commit()
            else:
                conn.rollback()
        return written

    def apply_supersession(
        self,
        superseded_old: MemoryCrystallizationV1,
        updated_new: MemoryCrystallizationV1,
        entries: list[GovernanceHistoryEntry],
        *,
        old_expected_statuses: Iterable[str] = ("active", "deprecated"),
    ) -> bool:
        """Atomically supersede old, link new, and record both audit rows."""
        with self._conn() as conn:
            with conn.cursor() as cur:
                written = self._upsert_on_cursor(
                    cur, superseded_old, expected_statuses=old_expected_statuses
                )
                if written:
                    self._upsert_on_cursor(cur, updated_new)
                    for entry in entries:
                        self._history_on_cursor(cur, entry)
            if written:
                conn.commit()
            else:
                conn.rollback()
        return written

    def merge_projection_refs(self, crystallization_id: str, ref_updates: dict[str, list[str]]) -> bool:
        """Union list fields into projection_refs without clobbering the doc.

        Row-locked read-modify-write so concurrent projections and governance
        transitions don't lose each other's writes. ref_updates maps
        projection_refs list-field names to ids to add. synced_at is bumped.
        """
        from datetime import datetime, timezone

        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT doc FROM memory_crystallizations WHERE crystallization_id = %s FOR UPDATE",
                    (crystallization_id,),
                )
                row = cur.fetchone()
                if row is None:
                    conn.rollback()
                    return False
                doc = row[0]
                refs = dict(doc.get("projection_refs") or {})
                for field, ids in ref_updates.items():
                    merged = sorted(set(refs.get(field) or []) | set(ids))
                    refs[field] = merged
                refs["synced_at"] = datetime.now(timezone.utc).isoformat()
                doc["projection_refs"] = refs
                cur.execute(
                    """
                    UPDATE memory_crystallizations
                    SET projection_refs = %s, doc = %s, updated_at = now()
                    WHERE crystallization_id = %s
                    """,
                    (json.dumps(refs), json.dumps(doc), crystallization_id),
                )
            conn.commit()
        return True

    def _upsert_on_cursor(
        self,
        cur,
        crystallization: MemoryCrystallizationV1,
        *,
        expected_statuses: Iterable[str] | None = None,
    ) -> bool:
        doc = crystallization.model_dump(mode="json")
        guard_sql = ""
        params: list[Any] = [
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
        ]
        if expected_statuses is not None:
            guard_sql = "WHERE memory_crystallizations.status = ANY(%s)"
            params.append(list(expected_statuses))
        cur.execute(
            f"""
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
            {guard_sql}
            """,
            params,
        )
        written = cur.rowcount > 0
        if written:
            self._rebuild_children(cur, crystallization)
        return written

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
        with self._conn() as conn:
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
        with self._conn() as conn:
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
        with self._conn() as conn:
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

    def _history_on_cursor(
        self,
        cur,
        entry: GovernanceHistoryEntry,
        *,
        before: dict[str, Any] | None = None,
        after: dict[str, Any] | None = None,
    ) -> None:
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

    def record_history(
        self,
        entry: GovernanceHistoryEntry,
        *,
        before: dict[str, Any] | None = None,
        after: dict[str, Any] | None = None,
    ) -> None:
        with self._conn() as conn:
            with conn.cursor() as cur:
                self._history_on_cursor(cur, entry, before=before, after=after)
            conn.commit()

    def list_history(self, crystallization_id: str, limit: int = 100) -> list[dict[str, Any]]:
        with self._conn() as conn:
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
        with self._conn() as conn:
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
        with self._conn() as conn:
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
