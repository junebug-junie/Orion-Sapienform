from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
from uuid import UUID, uuid4

import asyncpg

try:
    import psycopg2  # type: ignore
except Exception:  # pragma: no cover
    psycopg2 = None

from orion.core.contracts.memory_cards import (
    EvidenceItemV1,
    MemoryCardEdgeV1,
    MemoryCardHistoryEntryV1,
    MemoryCardV1,
    TimeHorizonV1,
)

logger = logging.getLogger("orion.memory_cards")

_SQL_PATH = Path(__file__).resolve().parents[3] / "services" / "orion-recall" / "sql" / "memory_cards.sql"


def apply_memory_cards_schema(dsn: str) -> None:
    """Apply Memory Cards DDL using psycopg2 (idempotent)."""
    if psycopg2 is None:
        raise RuntimeError("psycopg2 is required for apply_memory_cards_schema")
    if not _SQL_PATH.exists():
        raise FileNotFoundError(f"memory_cards.sql not found at {_SQL_PATH}")
    ddl = _SQL_PATH.read_text(encoding="utf-8")
    conn = psycopg2.connect(dsn)
    try:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(ddl)
    finally:
        conn.close()


def _memory_card_insert_params(card: MemoryCardV1) -> Dict[str, Any]:
    th = card.time_horizon.model_dump(mode="json") if card.time_horizon else None
    ev = [e.model_dump(mode="json") for e in card.evidence]
    return {
        "slug": card.slug,
        "types": card.types,
        "anchor_class": card.anchor_class,
        "status": card.status,
        "confidence": card.confidence,
        "sensitivity": card.sensitivity,
        "priority": card.priority,
        "visibility_scope": card.visibility_scope,
        "time_horizon": th,
        "provenance": card.provenance,
        "trust_source": card.trust_source,
        "project": card.project,
        "title": card.title,
        "summary": card.summary,
        "still_true": card.still_true,
        "anchors": card.anchors,
        "tags": card.tags,
        "evidence": ev,
        "subschema": dict(card.subschema or {}),
    }


def _row_to_card(row: asyncpg.Record) -> MemoryCardV1:
    th_raw = row["time_horizon"]

    th: Optional[TimeHorizonV1] = None
    if th_raw:
        if isinstance(th_raw, str):
            th_raw = json.loads(th_raw)
        if isinstance(th_raw, dict):
            th = TimeHorizonV1.model_validate(th_raw)
    ev_raw = row["evidence"]
    if isinstance(ev_raw, str):
        ev_raw = json.loads(ev_raw)
    evidence = [EvidenceItemV1.model_validate(x) for x in (ev_raw or [])]
    sub = row["subschema"]
    if isinstance(sub, str):
        sub = json.loads(sub)
    return MemoryCardV1(
        card_id=row["card_id"],
        slug=row["slug"],
        types=list(row["types"] or []),
        anchor_class=row["anchor_class"],
        status=row["status"],
        confidence=row["confidence"],
        sensitivity=row["sensitivity"],
        priority=row["priority"],
        visibility_scope=list(row["visibility_scope"] or []),
        time_horizon=th,
        provenance=row["provenance"],
        trust_source=row["trust_source"],
        project=row["project"],
        title=row["title"],
        summary=row["summary"],
        still_true=list(row["still_true"] or []) if row["still_true"] is not None else None,
        anchors=list(row["anchors"] or []) if row["anchors"] is not None else None,
        tags=list(row["tags"] or []) if row["tags"] is not None else None,
        evidence=evidence,
        subschema=dict(sub or {}),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


async def insert_card(
    pool: asyncpg.Pool,
    card: MemoryCardV1,
    *,
    actor: str,
    op: str = "create",
) -> UUID:
    data = _memory_card_insert_params(card)
    async with pool.acquire() as conn:
        async with conn.transaction():
            th_val = data["time_horizon"]
            row = await conn.fetchrow(
                """
                INSERT INTO memory_cards (
                  slug, types, anchor_class, status, confidence, sensitivity, priority,
                  visibility_scope, time_horizon, provenance, trust_source, project,
                  title, summary, still_true, anchors, tags, evidence, subschema
                ) VALUES (
                  $1,$2,$3,$4,$5,$6,$7,$8,
                  $9::jsonb,
                  $10,$11,$12,$13,$14,$15,$16,$17,$18::jsonb,$19::jsonb
                )
                RETURNING card_id
                """,
                data["slug"],
                data["types"],
                data["anchor_class"],
                data["status"],
                data["confidence"],
                data["sensitivity"],
                data["priority"],
                data["visibility_scope"],
                json.dumps(th_val) if th_val is not None else None,
                data["provenance"],
                data["trust_source"],
                data["project"],
                data["title"],
                data["summary"],
                data["still_true"],
                data["anchors"],
                data["tags"],
                json.dumps(data["evidence"]),
                json.dumps(data["subschema"]),
            )
            cid = row["card_id"]
            full = await conn.fetchrow("SELECT * FROM memory_cards WHERE card_id = $1", cid)
            after = json.loads(_row_to_card(full).model_dump_json())
            await conn.execute(
                """
                INSERT INTO memory_card_history (history_id, card_id, edge_id, op, actor, before, after)
                VALUES ($1, $2, NULL, $3, $4, NULL, $5::jsonb)
                """,
                uuid4(),
                cid,
                op,
                actor,
                json.dumps(after),
            )
            return cid


async def update_card(
    pool: asyncpg.Pool,
    card_id: str | UUID,
    patch: Dict[str, Any],
    *,
    actor: str,
) -> MemoryCardV1:
    cid = UUID(str(card_id)) if not isinstance(card_id, UUID) else card_id
    async with pool.acquire() as conn:
        async with conn.transaction():
            before_row = await conn.fetchrow("SELECT * FROM memory_cards WHERE card_id = $1", cid)
            if not before_row:
                raise LookupError("card not found")
            before_card = _row_to_card(before_row)
            before_json = json.loads(before_card.model_dump_json())
            allowed = {
                "slug",
                "types",
                "anchor_class",
                "status",
                "confidence",
                "sensitivity",
                "priority",
                "visibility_scope",
                "time_horizon",
                "provenance",
                "trust_source",
                "project",
                "title",
                "summary",
                "still_true",
                "anchors",
                "tags",
                "evidence",
                "subschema",
            }
            json_cols = {"evidence", "subschema", "time_horizon"}
            sets: List[str] = []
            args: List[Any] = []
            idx = 1
            for key, val in patch.items():
                if key not in allowed:
                    continue
                if key in json_cols and val is not None:
                    payload = json.dumps(val) if not isinstance(val, str) else val
                    sets.append(f"{key} = ${idx}::jsonb")
                    args.append(payload)
                else:
                    sets.append(f"{key} = ${idx}")
                    args.append(val)
                idx += 1
            if not sets:
                return before_card
            sets.append("updated_at = now()")
            args.append(cid)
            q = f"UPDATE memory_cards SET {', '.join(sets)} WHERE card_id = ${idx} RETURNING *"
            row = await conn.fetchrow(q, *args)
            after_card = _row_to_card(row)
            after_json = json.loads(after_card.model_dump_json())
            await conn.execute(
                """
                INSERT INTO memory_card_history (history_id, card_id, edge_id, op, actor, before, after)
                VALUES ($1, $2, NULL, 'update', $3, $4::jsonb, $5::jsonb)
                """,
                uuid4(),
                cid,
                actor,
                json.dumps(before_json),
                json.dumps(after_json),
            )
            return after_card


async def get_card(pool: asyncpg.Pool, card_id_or_slug: str) -> Optional[MemoryCardV1]:
    key = (card_id_or_slug or "").strip()
    if not key:
        return None
    async with pool.acquire() as conn:
        try:
            u = UUID(key)
            row = await conn.fetchrow("SELECT * FROM memory_cards WHERE card_id = $1 OR slug = $2", u, key)
        except ValueError:
            row = await conn.fetchrow("SELECT * FROM memory_cards WHERE slug = $1", key)
        if not row:
            return None
        return _row_to_card(row)


async def list_cards(
    pool: asyncpg.Pool,
    *,
    status: Optional[str] = None,
    types: Optional[List[str]] = None,
    anchor_class: Optional[str] = None,
    project: Optional[str] = None,
    priority: Optional[str] = None,
    limit: int = 200,
    offset: int = 0,
) -> List[MemoryCardV1]:
    where: List[str] = ["1=1"]
    args: List[Any] = []
    i = 1
    if status:
        where.append(f"status = ${i}")
        args.append(status)
        i += 1
    if types:
        where.append(f"types && ${i}::text[]")
        args.append(types)
        i += 1
    if anchor_class:
        where.append(f"anchor_class = ${i}")
        args.append(anchor_class)
        i += 1
    if project:
        where.append(f"project = ${i}")
        args.append(project)
        i += 1
    if priority:
        where.append(f"priority = ${i}")
        args.append(priority)
        i += 1
    args.extend([limit, offset])
    q = f"""
        SELECT * FROM memory_cards
        WHERE {' AND '.join(where)}
        ORDER BY updated_at DESC
        LIMIT ${i} OFFSET ${i + 1}
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(q, *args)
        return [_row_to_card(r) for r in rows]


async def _would_cycle_parent_child(
    conn: asyncpg.Connection,
    *,
    from_id: UUID,
    to_id: UUID,
    edge_type: str,
) -> bool:
    """Best-effort cycle check for hierarchy edges."""

    async def neighbors_down(parent: UUID) -> List[UUID]:
        rows = await conn.fetch(
            """
            SELECT to_card_id AS c FROM memory_card_edges
              WHERE from_card_id = $1 AND edge_type = 'parent_of'
            UNION ALL
            SELECT from_card_id AS c FROM memory_card_edges
              WHERE to_card_id = $1 AND edge_type = 'child_of'
            """,
            parent,
        )
        return [r["c"] for r in rows]

    async def neighbors_up(child: UUID) -> List[UUID]:
        rows = await conn.fetch(
            """
            SELECT from_card_id AS p FROM memory_card_edges
              WHERE to_card_id = $1 AND edge_type = 'parent_of'
            UNION ALL
            SELECT to_card_id AS p FROM memory_card_edges
              WHERE from_card_id = $1 AND edge_type = 'child_of'
            """,
            child,
        )
        return [r["p"] for r in rows]

    if edge_type == "parent_of":
        parent, child = from_id, to_id
        seen: set[UUID] = {child}
        frontier = [child]
        while frontier:
            cur = frontier.pop()
            for p in await neighbors_up(cur):
                if p == parent:
                    return True
                if p not in seen:
                    seen.add(p)
                    frontier.append(p)
    elif edge_type == "child_of":
        child, parent = from_id, to_id
        seen = {parent}
        frontier = [parent]
        while frontier:
            cur = frontier.pop()
            for c in await neighbors_down(cur):
                if c == child:
                    return True
                if c not in seen:
                    seen.add(c)
                    frontier.append(c)
    return False


async def add_edge(
    pool: asyncpg.Pool,
    edge: MemoryCardEdgeV1,
    *,
    actor: str,
) -> UUID:
    if edge.edge_type in ("parent_of", "child_of"):
        async with pool.acquire() as conn:
            if await _would_cycle_parent_child(
                conn,
                from_id=edge.from_card_id,
                to_id=edge.to_card_id,
                edge_type=str(edge.edge_type),
            ):
                raise ValueError("parent/child edge would create a cycle")
    meta = json.dumps(edge.metadata or {})
    async with pool.acquire() as conn:
        async with conn.transaction():
            row = await conn.fetchrow(
                """
                INSERT INTO memory_card_edges (from_card_id, to_card_id, edge_type, metadata)
                VALUES ($1,$2,$3,$4::jsonb)
                RETURNING edge_id
                """,
                edge.from_card_id,
                edge.to_card_id,
                edge.edge_type,
                meta,
            )
            eid = row["edge_id"]
            await conn.execute(
                """
                INSERT INTO memory_card_history (history_id, card_id, edge_id, op, actor, before, after)
                VALUES ($1, NULL, $2, 'edge_add', $3, NULL, $4::jsonb)
                """,
                uuid4(),
                eid,
                actor,
                json.dumps({"edge_id": str(eid), "from": str(edge.from_card_id), "to": str(edge.to_card_id), "type": edge.edge_type}),
            )
            return eid


async def remove_edge(pool: asyncpg.Pool, edge_id: str, *, actor: str) -> None:
    eid = UUID(str(edge_id))
    async with pool.acquire() as conn:
        async with conn.transaction():
            row = await conn.fetchrow("SELECT * FROM memory_card_edges WHERE edge_id = $1", eid)
            if not row:
                raise LookupError("edge not found")
            snap = dict(row)
            await conn.execute("DELETE FROM memory_card_edges WHERE edge_id = $1", eid)
            await conn.execute(
                """
                INSERT INTO memory_card_history (history_id, card_id, edge_id, op, actor, before, after)
                VALUES ($1, NULL, $2, 'edge_remove', $3, $4::jsonb, NULL)
                """,
                uuid4(),
                eid,
                actor,
                json.dumps(snap, default=str),
            )


async def list_edges(
    pool: asyncpg.Pool,
    *,
    card_id: str,
    direction: Literal["out", "in", "both"] = "both",
) -> List[MemoryCardEdgeV1]:
    cid = UUID(str(card_id))
    async with pool.acquire() as conn:
        rows: List[asyncpg.Record] = []
        if direction in ("out", "both"):
            rows.extend(await conn.fetch("SELECT * FROM memory_card_edges WHERE from_card_id = $1", cid))
        if direction in ("in", "both"):
            rows.extend(await conn.fetch("SELECT * FROM memory_card_edges WHERE to_card_id = $1", cid))
    out: List[MemoryCardEdgeV1] = []
    seen: set[UUID] = set()
    for r in rows:
        eid = r["edge_id"]
        if eid in seen:
            continue
        seen.add(eid)
        meta = r["metadata"]
        if isinstance(meta, str):
            meta = json.loads(meta)
        out.append(
            MemoryCardEdgeV1(
                edge_id=eid,
                from_card_id=r["from_card_id"],
                to_card_id=r["to_card_id"],
                edge_type=r["edge_type"],
                metadata=dict(meta or {}),
                created_at=r["created_at"],
            )
        )
    return out


async def list_history(
    pool: asyncpg.Pool,
    *,
    card_id: Optional[str] = None,
    edge_id: Optional[str] = None,
    limit: int = 500,
) -> List[MemoryCardHistoryEntryV1]:
    clauses: List[str] = []
    args: List[Any] = []
    i = 1
    if card_id:
        clauses.append(f"card_id = ${i}")
        args.append(UUID(str(card_id)))
        i += 1
    if edge_id:
        clauses.append(f"edge_id = ${i}")
        args.append(UUID(str(edge_id)))
        i += 1
    where = " AND ".join(clauses) if clauses else "TRUE"
    args.append(limit)
    lim_idx = len(args)
    q = f"SELECT * FROM memory_card_history WHERE {where} ORDER BY created_at DESC LIMIT ${lim_idx}"
    async with pool.acquire() as conn:
        rows = await conn.fetch(q, *args)
    res: List[MemoryCardHistoryEntryV1] = []
    for r in rows:
        res.append(
            MemoryCardHistoryEntryV1(
                history_id=r["history_id"],
                card_id=r["card_id"],
                edge_id=r["edge_id"],
                op=r["op"],
                actor=r["actor"],
                before=dict(r["before"]) if r["before"] else None,
                after=dict(r["after"]) if r["after"] else None,
                created_at=r["created_at"],
            )
        )
    return res


async def reverse_history(pool: asyncpg.Pool, history_id: str, *, actor: str) -> MemoryCardHistoryEntryV1:
    hid = UUID(str(history_id))
    async with pool.acquire() as conn:
        async with conn.transaction():
            h = await conn.fetchrow("SELECT * FROM memory_card_history WHERE history_id = $1", hid)
            if not h:
                raise LookupError("history not found")
            op = h["op"]
            if op == "create" and h["card_id"]:
                await conn.execute("DELETE FROM memory_cards WHERE card_id = $1", h["card_id"])
            elif op == "update" and h["card_id"] and h["before"]:
                b = h["before"]
                if isinstance(b, str):
                    b = json.loads(b)
                await conn.execute(
                    """
                    UPDATE memory_cards SET
                      slug=$1, types=$2, anchor_class=$3, status=$4, confidence=$5,
                      sensitivity=$6, priority=$7, visibility_scope=$8,
                      time_horizon=$9::jsonb, provenance=$10, trust_source=$11, project=$12,
                      title=$13, summary=$14, still_true=$15, anchors=$16, tags=$17,
                      evidence=$18::jsonb, subschema=$19::jsonb, updated_at=now()
                    WHERE card_id = $20
                    """,
                    b.get("slug"),
                    b.get("types"),
                    b.get("anchor_class"),
                    b.get("status"),
                    b.get("confidence"),
                    b.get("sensitivity"),
                    b.get("priority"),
                    b.get("visibility_scope"),
                    json.dumps(b.get("time_horizon")) if b.get("time_horizon") is not None else None,
                    b.get("provenance"),
                    b.get("trust_source"),
                    b.get("project"),
                    b.get("title"),
                    b.get("summary"),
                    b.get("still_true"),
                    b.get("anchors"),
                    b.get("tags"),
                    json.dumps(b.get("evidence") or []),
                    json.dumps(b.get("subschema") or {}),
                    h["card_id"],
                )
            elif op == "edge_add" and h["edge_id"]:
                await conn.execute("DELETE FROM memory_card_edges WHERE edge_id = $1", h["edge_id"])
            new_id = uuid4()
            await conn.execute(
                """
                INSERT INTO memory_card_history (history_id, card_id, edge_id, op, actor, before, after)
                VALUES ($1, $2, $3, 'reverse_auto_promotion', $4, $5::jsonb, NULL)
                """,
                new_id,
                h["card_id"],
                h["edge_id"],
                actor,
                json.dumps(dict(h), default=str),
            )
            row = await conn.fetchrow("SELECT * FROM memory_card_history WHERE history_id = $1", new_id)
    assert row is not None
    return MemoryCardHistoryEntryV1(
        history_id=row["history_id"],
        card_id=row["card_id"],
        edge_id=row["edge_id"],
        op=row["op"],
        actor=row["actor"],
        before=dict(row["before"]) if row["before"] else None,
        after=dict(row["after"]) if row["after"] else None,
        created_at=row["created_at"],
    )
