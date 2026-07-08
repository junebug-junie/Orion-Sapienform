from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any
from uuid import UUID

import asyncpg

from orion.memory.crystallization.recall_eligibility import eligible_for_recall
from orion.memory.crystallization.schemas import (
    CrystallizationClaimV1,
    CrystallizationDynamicsV1,
    CrystallizationEvidenceRefV1,
    CrystallizationGovernanceV1,
    CrystallizationLinkV1,
    CrystallizationProjectionRefsV1,
    MemoryCrystallizationV1,
    MemoryGrammarEnvelopeV1,
)

logger = logging.getLogger(__name__)


def _crystallizations_sql_path() -> Path:
    here = Path(__file__).resolve().parents[2] / "core" / "storage" / "sql" / "memory_crystallizations.sql"
    if here.is_file():
        return here
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "orion" / "core" / "storage" / "sql" / "memory_crystallizations.sql"


def apply_memory_crystallizations_schema(dsn: str) -> None:
    import psycopg2

    sql_path = _crystallizations_sql_path()
    if not sql_path.is_file():
        raise FileNotFoundError(f"memory_crystallizations DDL not found at {sql_path}")
    sql = sql_path.read_text(encoding="utf-8")
    with psycopg2.connect(dsn) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(sql)


def _jsonb(value: Any) -> str:
    return json.dumps(value)


def _parse_jsonb(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str) and value.strip():
        return json.loads(value)
    return None


def _row_to_crystallization(
    row: asyncpg.Record,
    *,
    claims: list[CrystallizationClaimV1],
    evidence: list[CrystallizationEvidenceRefV1],
    links: list[CrystallizationLinkV1],
) -> MemoryCrystallizationV1:
    gov_raw = _parse_jsonb(row["governance"]) or {}
    proj_raw = _parse_jsonb(row["projection_refs"]) or {}
    grammar_raw = _parse_jsonb(row["grammar_envelope"]) or {}
    dynamics_raw = _parse_jsonb(row.get("dynamics")) or {}

    return MemoryCrystallizationV1(
        crystallization_id=str(row["crystallization_id"]),
        kind=row["kind"],
        subject=row["subject"],
        summary=row["summary"],
        status=row["status"],
        confidence=row["confidence"],
        salience=float(row["salience"]),
        dynamics=CrystallizationDynamicsV1.model_validate(dynamics_raw),
        scope=list(row["scope"] or []),
        tags=list(row["tags"] or []),
        claims=claims,
        evidence=evidence,
        source_card_ids=list(row["source_card_ids"] or []),
        source_grammar_event_ids=list(row["source_grammar_event_ids"] or []),
        source_atom_ids=list(row["source_atom_ids"] or []),
        grammar_envelope=MemoryGrammarEnvelopeV1.model_validate(grammar_raw),
        planning_effects=list(row["planning_effects"] or []),
        retrieval_affordances=list(row["retrieval_affordances"] or []),
        links=links,
        projection_refs=CrystallizationProjectionRefsV1.model_validate(proj_raw),
        governance=CrystallizationGovernanceV1.model_validate(gov_raw),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


async def _load_children(
    pool: asyncpg.Pool,
    crystallization_id: str,
) -> tuple[list[CrystallizationClaimV1], list[CrystallizationEvidenceRefV1], list[CrystallizationLinkV1]]:
    async with pool.acquire() as conn:
        claim_rows = await conn.fetch(
            """
            SELECT claim_id, claim, status, confidence, evidence_ref_ids
            FROM memory_crystallization_claims
            WHERE crystallization_id = $1::uuid
            ORDER BY created_at
            """,
            crystallization_id,
        )
        source_rows = await conn.fetch(
            """
            SELECT source_kind, source_id, excerpt, strength, note
            FROM memory_crystallization_sources
            WHERE crystallization_id = $1::uuid
            ORDER BY created_at
            """,
            crystallization_id,
        )
        link_rows = await conn.fetch(
            """
            SELECT to_crystallization_id, relation, confidence, note
            FROM memory_crystallization_links
            WHERE from_crystallization_id = $1::uuid
            ORDER BY created_at
            """,
            crystallization_id,
        )

    claims = [
        CrystallizationClaimV1(
            claim_id=str(r["claim_id"]),
            claim=r["claim"],
            status=r["status"],
            confidence=r["confidence"],
            evidence_refs=list(r["evidence_ref_ids"] or []),
        )
        for r in claim_rows
    ]
    evidence = [
        CrystallizationEvidenceRefV1(
            source_kind=r["source_kind"],
            source_id=r["source_id"],
            excerpt=r["excerpt"],
            strength=float(r["strength"]),
            note=r["note"],
        )
        for r in source_rows
    ]
    links = [
        CrystallizationLinkV1(
            target_crystallization_id=str(r["to_crystallization_id"]),
            relation=r["relation"],
            confidence=float(r["confidence"]),
            note=r["note"],
        )
        for r in link_rows
    ]
    return claims, evidence, links


async def insert_crystallization(pool: asyncpg.Pool, crystallization: MemoryCrystallizationV1) -> str:
    cid = crystallization.crystallization_id
    try:
        UUID(str(cid).replace("crys_", ""))
        db_id = str(cid).replace("crys_", "")
        if len(db_id) == 32:
            db_uuid = f"{db_id[:8]}-{db_id[8:12]}-{db_id[12:16]}-{db_id[16:20]}-{db_id[20:]}"
        else:
            db_uuid = cid if "-" in str(cid) else str(UUID(int=0))
    except ValueError:
        db_uuid = None

    async with pool.acquire() as conn:
        async with conn.transaction():
            if db_uuid:
                row = await conn.fetchrow(
                    """
                    INSERT INTO memory_crystallizations (
                        crystallization_id, kind, subject, summary, status, confidence, salience,
                        dynamics, scope, tags, grammar_envelope, planning_effects, retrieval_affordances,
                        governance, projection_refs, source_card_ids, source_grammar_event_ids,
                        source_atom_ids, created_at, updated_at
                    ) VALUES (
                        $1::uuid, $2, $3, $4, $5, $6, $7,
                        $8::jsonb, $9, $10, $11::jsonb, $12, $13,
                        $14::jsonb, $15::jsonb, $16, $17,
                        $18, $19, $20
                    )
                    RETURNING crystallization_id
                    """,
                    db_uuid,
                    crystallization.kind,
                    crystallization.subject,
                    crystallization.summary,
                    crystallization.status,
                    crystallization.confidence,
                    crystallization.salience,
                    _jsonb(crystallization.dynamics.model_dump(mode="json")),
                    crystallization.scope,
                    crystallization.tags,
                    _jsonb(crystallization.grammar_envelope.model_dump(mode="json")),
                    crystallization.planning_effects,
                    crystallization.retrieval_affordances,
                    _jsonb(crystallization.governance.model_dump(mode="json")),
                    _jsonb(crystallization.projection_refs.model_dump(mode="json")),
                    crystallization.source_card_ids,
                    crystallization.source_grammar_event_ids,
                    crystallization.source_atom_ids,
                    crystallization.created_at,
                    crystallization.updated_at,
                )
            else:
                row = await conn.fetchrow(
                    """
                    INSERT INTO memory_crystallizations (
                        kind, subject, summary, status, confidence, salience,
                        dynamics, scope, tags, grammar_envelope, planning_effects, retrieval_affordances,
                        governance, projection_refs, source_card_ids, source_grammar_event_ids,
                        source_atom_ids, created_at, updated_at
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6,
                        $7::jsonb, $8, $9, $10::jsonb, $11, $12,
                        $13::jsonb, $14::jsonb, $15, $16,
                        $17, $18, $19
                    )
                    RETURNING crystallization_id
                    """,
                    crystallization.kind,
                    crystallization.subject,
                    crystallization.summary,
                    crystallization.status,
                    crystallization.confidence,
                    crystallization.salience,
                    _jsonb(crystallization.dynamics.model_dump(mode="json")),
                    crystallization.scope,
                    crystallization.tags,
                    _jsonb(crystallization.grammar_envelope.model_dump(mode="json")),
                    crystallization.planning_effects,
                    crystallization.retrieval_affordances,
                    _jsonb(crystallization.governance.model_dump(mode="json")),
                    _jsonb(crystallization.projection_refs.model_dump(mode="json")),
                    crystallization.source_card_ids,
                    crystallization.source_grammar_event_ids,
                    crystallization.source_atom_ids,
                    crystallization.created_at,
                    crystallization.updated_at,
                )

            stored_id = str(row["crystallization_id"])

            for claim in crystallization.claims:
                await conn.execute(
                    """
                    INSERT INTO memory_crystallization_claims
                        (crystallization_id, claim, status, confidence, evidence_ref_ids)
                    VALUES ($1::uuid, $2, $3, $4, $5)
                    """,
                    stored_id,
                    claim.claim,
                    claim.status,
                    claim.confidence,
                    claim.evidence_refs,
                )

            for ev in crystallization.evidence:
                await conn.execute(
                    """
                    INSERT INTO memory_crystallization_sources
                        (crystallization_id, source_kind, source_id, excerpt, strength, note)
                    VALUES ($1::uuid, $2, $3, $4, $5, $6)
                    """,
                    stored_id,
                    ev.source_kind,
                    ev.source_id,
                    ev.excerpt,
                    ev.strength,
                    ev.note,
                )

            for link in crystallization.links:
                target = link.target_crystallization_id
                try:
                    target_uuid = target.replace("crys_", "")
                    if len(target_uuid) == 32:
                        target = f"{target_uuid[:8]}-{target_uuid[8:12]}-{target_uuid[12:16]}-{target_uuid[16:20]}-{target_uuid[20:]}"
                except Exception:
                    pass
                await conn.execute(
                    """
                    INSERT INTO memory_crystallization_links
                        (from_crystallization_id, to_crystallization_id, relation, confidence, note)
                    VALUES ($1::uuid, $2::uuid, $3, $4, $5)
                    ON CONFLICT (from_crystallization_id, to_crystallization_id, relation) DO NOTHING
                    """,
                    stored_id,
                    target,
                    link.relation,
                    link.confidence,
                    link.note,
                )

    return stored_id


async def update_crystallization(pool: asyncpg.Pool, crystallization: MemoryCrystallizationV1) -> None:
    cid = str(crystallization.crystallization_id)
    if cid.startswith("crys_"):
        hex_id = cid.replace("crys_", "")
        if len(hex_id) == 32:
            cid = f"{hex_id[:8]}-{hex_id[8:12]}-{hex_id[12:16]}-{hex_id[16:20]}-{hex_id[20:]}"

    async with pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE memory_crystallizations SET
                kind = $2, subject = $3, summary = $4, status = $5, confidence = $6,
                salience = $7, dynamics = $8::jsonb, scope = $9, tags = $10,
                grammar_envelope = $11::jsonb, planning_effects = $12,
                retrieval_affordances = $13, governance = $14::jsonb,
                projection_refs = $15::jsonb, source_card_ids = $16,
                source_grammar_event_ids = $17, source_atom_ids = $18, updated_at = $19
            WHERE crystallization_id = $1::uuid
            """,
            cid,
            crystallization.kind,
            crystallization.subject,
            crystallization.summary,
            crystallization.status,
            crystallization.confidence,
            crystallization.salience,
            _jsonb(crystallization.dynamics.model_dump(mode="json")),
            crystallization.scope,
            crystallization.tags,
            _jsonb(crystallization.grammar_envelope.model_dump(mode="json")),
            crystallization.planning_effects,
            crystallization.retrieval_affordances,
            _jsonb(crystallization.governance.model_dump(mode="json")),
            _jsonb(crystallization.projection_refs.model_dump(mode="json")),
            crystallization.source_card_ids,
            crystallization.source_grammar_event_ids,
            crystallization.source_atom_ids,
            crystallization.updated_at,
        )


async def get_crystallization(pool: asyncpg.Pool, crystallization_id: str) -> MemoryCrystallizationV1 | None:
    cid = crystallization_id
    if cid.startswith("crys_"):
        hex_id = cid.replace("crys_", "")
        if len(hex_id) == 32:
            cid = f"{hex_id[:8]}-{hex_id[8:12]}-{hex_id[12:16]}-{hex_id[16:20]}-{hex_id[20:]}"

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM memory_crystallizations WHERE crystallization_id = $1::uuid",
            cid,
        )
    if not row:
        return None

    claims, evidence, links = await _load_children(pool, str(row["crystallization_id"]))
    return _row_to_crystallization(row, claims=claims, evidence=evidence, links=links)


async def list_crystallizations(
    pool: asyncpg.Pool,
    *,
    status: str | None = None,
    kind: str | None = None,
    limit: int = 200,
    offset: int = 0,
) -> list[MemoryCrystallizationV1]:
    clauses = ["1=1"]
    args: list[Any] = []
    idx = 1
    if status:
        clauses.append(f"status = ${idx}")
        args.append(status)
        idx += 1
    if kind:
        clauses.append(f"kind = ${idx}")
        args.append(kind)
        idx += 1

    sql = f"""
        SELECT * FROM memory_crystallizations
        WHERE {' AND '.join(clauses)}
        ORDER BY salience DESC, updated_at DESC
        LIMIT ${idx} OFFSET ${idx + 1}
    """
    args.extend([limit, offset])

    async with pool.acquire() as conn:
        rows = await conn.fetch(sql, *args)

    out: list[MemoryCrystallizationV1] = []
    for row in rows:
        claims, evidence, links = await _load_children(pool, str(row["crystallization_id"]))
        out.append(_row_to_crystallization(row, claims=claims, evidence=evidence, links=links))
    return out


async def count_eligible_active(pool: asyncpg.Pool, *, floor: float | None = None) -> int:
    rows = await list_crystallizations(pool, status="active", limit=500)
    return sum(1 for r in rows if eligible_for_recall(r, floor=floor))


async def insert_history(
    pool: asyncpg.Pool,
    *,
    crystallization_id: str,
    op: str,
    actor: str,
    before: dict | None,
    after: dict | None,
    reason: str | None = None,
) -> None:
    cid = crystallization_id
    if cid.startswith("crys_"):
        hex_id = cid.replace("crys_", "")
        if len(hex_id) == 32:
            cid = f"{hex_id[:8]}-{hex_id[8:12]}-{hex_id[12:16]}-{hex_id[16:20]}-{hex_id[20:]}"

    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO memory_crystallization_history
                (crystallization_id, op, actor, before, after, reason)
            VALUES ($1::uuid, $2, $3, $4::jsonb, $5::jsonb, $6)
            """,
            cid,
            op,
            actor,
            _jsonb(before) if before else None,
            _jsonb(after) if after else None,
            reason,
        )


async def insert_retrieval_event(
    pool: asyncpg.Pool,
    *,
    query: str,
    task_type: str | None,
    project_id: str | None,
    session_id: str | None,
    crystallization_ids: list[str],
    card_refs: list[str],
    trace: dict,
) -> str:
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO memory_crystallization_retrieval_events
                (query, task_type, project_id, session_id, crystallization_ids, card_refs, trace)
            VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb)
            RETURNING retrieval_event_id
            """,
            query,
            task_type,
            project_id,
            session_id,
            crystallization_ids,
            card_refs,
            _jsonb(trace),
        )
    return str(row["retrieval_event_id"])
