from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple
from uuid import UUID, uuid4

import asyncpg
import psycopg2

from orion.core.contracts.memory_cards import (
    MemoryCardCreateV1,
    MemoryCardEdgeCreateV1,
    MemoryCardEdgeV1,
    MemoryCardHistoryEntryV1,
    MemoryCardPatchV1,
    MemoryCardV1,
    TimeHorizonV1,
    EvidenceItemV1,
)

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[4]
_MEMORY_CARDS_SQL = _REPO_ROOT / "services" / "orion-recall" / "sql" / "memory_cards.sql"


def apply_memory_cards_schema(dsn: str) -> None:
    """Apply idempotent DDL using psycopg2 (sync bootstrap)."""
    sql = _MEMORY_CARDS_SQL.read_text(encoding="utf-8")
    with psycopg2.connect(dsn) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(sql)


def _slugify(title: str) -> str:
    base = re.sub(r"[^a-z0-9]+", "-", title.lower().strip())[:80].strip("-") or "card"
    return f"{base}-{uuid4().hex[:8]}"


def _row_to_card(row: asyncpg.Record) -> MemoryCardV1:
    th = row["time_horizon"]
    time_horizon = TimeHorizonV1.model_validate(th) if isinstance(th, dict) else None
    ev = row["evidence"] or []
    evidence = [EvidenceItemV1.model_validate(x) for x in ev] if isinstance(ev, list) else []
    sub = row["subschema"]
    subschema = dict(sub) if isinstance(sub, dict) else {}
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
        time_horizon=time_horizon,
        provenance=row["provenance"],
        trust_source=row["trust_source"],
        project=row["project"],
        title=row["title"],
        summary=row["summary"],
        still_true=list(row["still_true"] or []) if row["still_true"] else None,
        anchors=list(row["anchors"] or []) if row["anchors"] else None,
        tags=list(row["tags"] or []) if row["tags"] else None,
        evidence=evidence,
        subschema=subschema,
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _row_to_edge(row: asyncpg.Record) -> MemoryCardEdgeV1:
    return MemoryCardEdgeV1(
        edge_id=row["edge_id"],
        from_card_id=row["from_card_id"],
        to_card_id=row["to_card_id"],
        edge_type=row["edge_type"],
        metadata=dict(row["metadata"] or {}),
        created_at=row["created_at"],
    )


def _row_to_history(row: asyncpg.Record) -> MemoryCardHistoryEntryV1:
    return MemoryCardHistoryEntryV1(
        history_id=row["history_id"],
        card_id=row["card_id"],
        edge_id=row["edge_id"],
        op=row["op"],
        actor=row["actor"],
        before=dict(row["before"]) if isinstance(row["before"], dict) else None,
        after=dict(row["after"]) if isinstance(row["after"], dict) else None,
        created_at=row["created_at"],
    )


async def _resolve_card_id(conn: asyncpg.Connection, card_id_or_slug: str) -> Optional[UUID]:
    s = str(card_id_or_slug).strip()
    if not s:
        return None
    try:
        return UUID(s)
    except ValueError:
        row = await conn.fetchrow("SELECT card_id FROM memory_cards WHERE slug = $1", s)
        return row["card_id"] if row else None


def _hierarchy_parent_map(
    rows: List[asyncpg.Record], *, extra: Optional[Tuple[UUID, UUID, str]] = None
) -> Dict[UUID, UUID]:
    """Map child_id -> parent_id for parent_of / child_of edges (last edge wins per child)."""
    parents: Dict[UUID, UUID] = {}
    for r in rows:
        f, t, et = r["from_card_id"], r["to_card_id"], r["edge_type"]
        if et == "parent_of":
            parents[t] = f
        elif et == "child_of":
            parents[f] = t
    if extra:
        f, t, et = extra
        if et == "parent_of":
            parents[t] = f
        elif et == "child_of":
            parents[f] = t
    return parents


async def _hierarchy_edge_would_cycle(
    conn: asyncpg.Connection, *, from_card_id: UUID, to_card_id: UUID, edge_type: str
) -> bool:
    if edge_type not in {"parent_of", "child_of"}:
        return False
    if from_card_id == to_card_id:
        return True
    rows = await conn.fetch(
        """
        SELECT from_card_id, to_card_id, edge_type
        FROM memory_card_edges
        WHERE edge_type IN ('parent_of', 'child_of')
        """
    )
    proposed = (from_card_id, to_card_id, edge_type)
    parents = _hierarchy_parent_map(list(rows), extra=proposed)
    if edge_type == "parent_of":
        parent, child = from_card_id, to_card_id
    else:
        parent, child = to_card_id, from_card_id
    cur: Optional[UUID] = parent
    seen: set[UUID] = set()
    while cur is not None:
        if cur == child:
            return True
        if cur in seen:
            return True
        seen.add(cur)
        cur = parents.get(cur)
    return False


async def insert_card(pool: asyncpg.Pool, card: MemoryCardCreateV1, *, actor: str, op: str = "create") -> UUID:
    slug = (card.slug or "").strip() or _slugify(card.title)
    body = card.model_dump(mode="json", exclude={"slug"})
    body["slug"] = slug
    th_val = card.time_horizon.model_dump(mode="json") if card.time_horizon else None
    evidence_val = [e.model_dump(mode="json") for e in card.evidence]
    sub_val = dict(card.subschema or {})
    async with pool.acquire() as conn:
        async with conn.transaction():
            row = await conn.fetchrow(
                """
                INSERT INTO memory_cards (
                  slug, types, anchor_class, status, confidence, sensitivity, priority,
                  visibility_scope, time_horizon, provenance, trust_source, project,
                  title, summary, still_true, anchors, tags, evidence, subschema
                ) VALUES (
                  $1,$2,$3,$4,$5,$6,$7,$8,$9::jsonb,$10,$11,$12,$13,$14,$15,$16,$17,$18::jsonb,$19::jsonb
                )
                RETURNING card_id
                """,
                body["slug"],
                body["types"],
                body.get("anchor_class"),
                body["status"],
                body["confidence"],
                body["sensitivity"],
                body["priority"],
                body["visibility_scope"],
                th_val,
                body["provenance"],
                body.get("trust_source"),
                body.get("project"),
                body["title"],
                body["summary"],
                body.get("still_true"),
                body.get("anchors"),
                body.get("tags"),
                evidence_val,
                sub_val,
            )
            cid = row["card_id"]
            snap = await conn.fetchrow("SELECT to_jsonb(mc.*) AS j FROM memory_cards mc WHERE mc.card_id = $1", cid)
            await conn.execute(
                """
                INSERT INTO memory_card_history (card_id, edge_id, op, actor, "before", "after")
                VALUES ($1, NULL, $2, $3, NULL, $4::jsonb)
                """,
                cid,
                op,
                actor,
                json.dumps(snap["j"]),
            )
            return cid


async def get_card(pool: asyncpg.Pool, card_id_or_slug: str) -> Optional[MemoryCardV1]:
    async with pool.acquire() as conn:
        cid = await _resolve_card_id(conn, card_id_or_slug)
        if cid is None:
            return None
        row = await conn.fetchrow("SELECT * FROM memory_cards WHERE card_id = $1", cid)
        return _row_to_card(row) if row else None


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
    limit = max(1, min(int(limit), 500))
    offset = max(0, min(int(offset), 100_000))
    clauses: List[str] = []
    args: List[Any] = []
    i = 1
    if status:
        clauses.append(f"status = ${i}")
        args.append(status)
        i += 1
    if types:
        clauses.append(f"types && ${i}::text[]")
        args.append(types)
        i += 1
    if anchor_class:
        clauses.append(f"anchor_class = ${i}")
        args.append(anchor_class)
        i += 1
    if project:
        clauses.append(f"project = ${i}")
        args.append(project)
        i += 1
    if priority:
        clauses.append(f"priority = ${i}")
        args.append(priority)
        i += 1
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    args.extend([limit, offset])
    sql = f"SELECT * FROM memory_cards {where} ORDER BY updated_at DESC LIMIT ${i} OFFSET ${i + 1}"
    async with pool.acquire() as conn:
        rows = await conn.fetch(sql, *args)
        return [_row_to_card(r) for r in rows]


_ALLOWED_PATCH_COLS = frozenset(
    {
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
)


def _patch_to_columns(patch: MemoryCardPatchV1) -> Dict[str, Any]:
    raw = patch.model_dump(exclude_none=True, mode="json")
    data = {k: v for k, v in raw.items() if k in _ALLOWED_PATCH_COLS}
    return data


async def update_card(
    pool: asyncpg.Pool, card_id_or_slug: str, patch: MemoryCardPatchV1, *, actor: str
) -> Optional[MemoryCardV1]:
    cols = _patch_to_columns(patch)
    if not cols:
        return await get_card(pool, card_id_or_slug)
    async with pool.acquire() as conn:
        cid = await _resolve_card_id(conn, card_id_or_slug)
        if cid is None:
            return None
        before = await conn.fetchrow("SELECT to_jsonb(mc.*) AS j FROM memory_cards mc WHERE mc.card_id = $1", cid)
        if not before:
            return None
        set_parts = []
        args: List[Any] = []
        idx = 1
        jsonb_cols = {"time_horizon", "evidence", "subschema"}
        for key, val in cols.items():
            cast = "::jsonb" if key in jsonb_cols else ""
            set_parts.append(f"{key} = ${idx}{cast}")
            args.append(val)
            idx += 1
        set_parts.append("updated_at = now()")
        args.append(cid)
        sql = f"UPDATE memory_cards SET {', '.join(set_parts)} WHERE card_id = ${idx} RETURNING *"
        async with conn.transaction():
            row = await conn.fetchrow(sql, *args)
            after = await conn.fetchrow("SELECT to_jsonb(mc.*) AS j FROM memory_cards mc WHERE mc.card_id = $1", cid)
            await conn.execute(
                """
                INSERT INTO memory_card_history (card_id, edge_id, op, actor, "before", "after")
                VALUES ($1, NULL, 'update', $2, $3::jsonb, $4::jsonb)
                """,
                cid,
                actor,
                json.dumps(before["j"]),
                json.dumps(after["j"]),
            )
        return _row_to_card(row) if row else None


async def change_card_status(
    pool: asyncpg.Pool,
    card_id: str,
    *,
    new_status: str,
    actor: str,
    reason: Optional[str] = None,
) -> Optional[MemoryCardV1]:
    async with pool.acquire() as conn:
        cid = await _resolve_card_id(conn, card_id)
        if cid is None:
            return None
        before = await conn.fetchrow("SELECT to_jsonb(mc.*) AS j FROM memory_cards mc WHERE mc.card_id = $1", cid)
        async with conn.transaction():
            row = await conn.fetchrow(
                "UPDATE memory_cards SET status = $2, updated_at = now() WHERE card_id = $1 RETURNING *",
                cid,
                new_status,
            )
            after = await conn.fetchrow("SELECT to_jsonb(mc.*) AS j FROM memory_cards mc WHERE mc.card_id = $1", cid)
            meta_after = {"card": after["j"], "reason": reason}
            meta_before = {"card": before["j"]}
            await conn.execute(
                """
                INSERT INTO memory_card_history (card_id, edge_id, op, actor, "before", "after")
                VALUES ($1, NULL, 'status_change', $2, $3::jsonb, $4::jsonb)
                """,
                cid,
                actor,
                json.dumps(meta_before),
                json.dumps(meta_after),
            )
        return _row_to_card(row) if row else None


async def add_edge(pool: asyncpg.Pool, edge: MemoryCardEdgeCreateV1, *, actor: str) -> MemoryCardEdgeV1:
    async with pool.acquire() as conn:
        if await _hierarchy_edge_would_cycle(
            conn, from_card_id=edge.from_card_id, to_card_id=edge.to_card_id, edge_type=edge.edge_type
        ):
            raise ValueError("hierarchy_cycle")
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
                json.dumps(edge.metadata or {}),
            )
            eid = row["edge_id"]
            snap = await conn.fetchrow("SELECT to_jsonb(e.*) AS j FROM memory_card_edges e WHERE e.edge_id = $1", eid)
            await conn.execute(
                """
                INSERT INTO memory_card_history (card_id, edge_id, op, actor, "before", "after")
                VALUES ($1, $2, 'edge_add', $3, NULL, $4::jsonb)
                """,
                edge.from_card_id,
                eid,
                actor,
                json.dumps(snap["j"]),
            )
            er = await conn.fetchrow("SELECT * FROM memory_card_edges WHERE edge_id = $1", eid)
            assert er is not None
            return _row_to_edge(er)


async def remove_edge(pool: asyncpg.Pool, edge_id: str, *, actor: str) -> None:
    e_uuid = UUID(str(edge_id).strip())
    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT * FROM memory_card_edges WHERE edge_id = $1", e_uuid)
        if not row:
            return
        before = await conn.fetchrow("SELECT to_jsonb(e.*) AS j FROM memory_card_edges e WHERE e.edge_id = $1", e_uuid)
        async with conn.transaction():
            await conn.execute("DELETE FROM memory_card_edges WHERE edge_id = $1", e_uuid)
            await conn.execute(
                """
                INSERT INTO memory_card_history (card_id, edge_id, op, actor, "before", "after")
                VALUES ($1, $2, 'edge_remove', $3, $4::jsonb, NULL)
                """,
                row["from_card_id"],
                e_uuid,
                actor,
                json.dumps(before["j"]),
            )


async def list_edges(
    pool: asyncpg.Pool, *, card_id: str, direction: Literal["out", "in", "both"] = "both"
) -> List[MemoryCardEdgeV1]:
    async with pool.acquire() as conn:
        cid = await _resolve_card_id(conn, card_id)
        if cid is None:
            return []
        if direction == "out":
            rows = await conn.fetch(
                "SELECT * FROM memory_card_edges WHERE from_card_id = $1 ORDER BY created_at DESC", cid
            )
        elif direction == "in":
            rows = await conn.fetch(
                "SELECT * FROM memory_card_edges WHERE to_card_id = $1 ORDER BY created_at DESC", cid
            )
        else:
            rows = await conn.fetch(
                """
                SELECT * FROM memory_card_edges
                WHERE from_card_id = $1 OR to_card_id = $1
                ORDER BY created_at DESC
                """,
                cid,
            )
        return [_row_to_edge(r) for r in rows]


async def list_history(
    pool: asyncpg.Pool,
    *,
    card_id: Optional[str] = None,
    edge_id: Optional[str] = None,
    limit: int = 500,
    offset: int = 0,
) -> List[MemoryCardHistoryEntryV1]:
    limit = max(1, min(int(limit), 500))
    offset = max(0, min(int(offset), 100_000))
    async with pool.acquire() as conn:
        clauses = []
        args: List[Any] = []
        idx = 1
        if card_id:
            cid = await _resolve_card_id(conn, card_id)
            if cid is None:
                return []
            clauses.append(f"card_id = ${idx}")
            args.append(cid)
            idx += 1
        if edge_id:
            clauses.append(f"edge_id = ${idx}")
            args.append(UUID(str(edge_id).strip()))
            idx += 1
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        args.extend([limit, offset])
        sql = f"""
            SELECT * FROM memory_card_history
            {where}
            ORDER BY created_at DESC
            LIMIT ${idx} OFFSET ${idx + 1}
        """
        rows = await conn.fetch(sql, *args)
        return [_row_to_history(r) for r in rows]


async def reverse_history(pool: asyncpg.Pool, history_id: str, *, actor: str) -> MemoryCardHistoryEntryV1:
    hid = UUID(str(history_id).strip())
    async with pool.acquire() as conn:
        entry = await conn.fetchrow("SELECT * FROM memory_card_history WHERE history_id = $1", hid)
        if not entry:
            raise LookupError("history_not_found")
        op = entry["op"]
        async with conn.transaction():
            if op == "edge_add":
                edge_j = entry["after"]
                if not isinstance(edge_j, dict) or "edge_id" not in edge_j:
                    raise ValueError("reverse_malformed_history")
                eid = UUID(str(edge_j["edge_id"]))
                await conn.execute("DELETE FROM memory_card_edges WHERE edge_id = $1", eid)
                rev = await conn.fetchrow(
                    """
                    INSERT INTO memory_card_history (card_id, edge_id, op, actor, "before", "after")
                    VALUES ($1, $2, 'reverse_auto_promotion', $3, $4::jsonb, NULL)
                    RETURNING *
                    """,
                    entry["card_id"],
                    eid,
                    actor,
                    json.dumps({"reversed_history_id": str(hid)}),
                )
                return _row_to_history(rev)
            if op == "update":
                before = entry["before"]
                if not isinstance(before, dict):
                    raise ValueError("reverse_malformed_history")
                card_j = before.get("card_id")
                if card_j is None:
                    raise ValueError("reverse_malformed_history")
                cid = UUID(str(card_j))
                # Restore full row from before snapshot
                b = before
                await conn.execute(
                    """
                    UPDATE memory_cards SET
                      slug=$2, types=$3, anchor_class=$4, status=$5, confidence=$6,
                      sensitivity=$7, priority=$8, visibility_scope=$9,
                      time_horizon=$10::jsonb, provenance=$11, trust_source=$12,
                      project=$13, title=$14, summary=$15, still_true=$16,
                      anchors=$17, tags=$18, evidence=$19::jsonb, subschema=$20::jsonb,
                      updated_at=now()
                    WHERE card_id=$1
                    """,
                    cid,
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
                )
                rev = await conn.fetchrow(
                    """
                    INSERT INTO memory_card_history (card_id, edge_id, op, actor, "before", "after")
                    VALUES ($1, NULL, 'reverse_auto_promotion', $2, $3::jsonb, NULL)
                    RETURNING *
                    """,
                    cid,
                    actor,
                    json.dumps({"reversed_history_id": str(hid)}),
                )
                return _row_to_history(rev)
            if op == "status_change":
                before = entry["before"]
                if not isinstance(before, dict) or not isinstance(before.get("card"), dict):
                    raise ValueError("reverse_malformed_history")
                card = before["card"]
                cid = UUID(str(card["card_id"]))
                await conn.execute(
                    "UPDATE memory_cards SET status = $2, updated_at = now() WHERE card_id = $1",
                    cid,
                    card.get("status"),
                )
                rev = await conn.fetchrow(
                    """
                    INSERT INTO memory_card_history (card_id, edge_id, op, actor, "before", "after")
                    VALUES ($1, NULL, 'reverse_auto_promotion', $2, $3::jsonb, NULL)
                    RETURNING *
                    """,
                    cid,
                    actor,
                    json.dumps({"reversed_history_id": str(hid)}),
                )
                return _row_to_history(rev)
        raise ValueError("reverse_unsupported_op")


async def neighborhood(
    pool: asyncpg.Pool, card_id_or_slug: str, *, hops: int = 1
) -> Dict[str, Any]:
    hops = max(1, min(int(hops), 3))
    async with pool.acquire() as conn:
        root_id = await _resolve_card_id(conn, card_id_or_slug)
        if root_id is None:
            return {"card_id": None, "hops": hops, "by_edge_type": {}}
        frontier = {root_id}
        seen = {root_id}
        by_type: Dict[str, List[Dict[str, Any]]] = {}
        for _ in range(hops):
            next_frontier: set[UUID] = set()
            for cid in frontier:
                erows = await conn.fetch(
                    "SELECT * FROM memory_card_edges WHERE from_card_id = $1 OR to_card_id = $1",
                    cid,
                )
                for e in erows:
                    nbr = e["to_card_id"] if e["from_card_id"] == cid else e["from_card_id"]
                    if nbr in seen:
                        continue
                    seen.add(nbr)
                    next_frontier.add(nbr)
                    mc = await conn.fetchrow("SELECT slug, title, summary, status FROM memory_cards WHERE card_id = $1", nbr)
                    if not mc:
                        continue
                    et = e["edge_type"]
                    by_type.setdefault(et, []).append(
                        {
                            "card_id": str(nbr),
                            "slug": mc["slug"],
                            "title": mc["title"],
                            "summary": mc["summary"],
                            "status": mc["status"],
                            "via": "out" if e["from_card_id"] == cid else "in",
                        }
                    )
            frontier = next_frontier
        return {"card_id": str(root_id), "hops": hops, "by_edge_type": by_type}
