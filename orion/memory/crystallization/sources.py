from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import asyncpg

from orion.core.storage import memory_cards as mc_dal
from orion.memory.crystallization.schemas import CrystallizationEvidenceRefV1, MemoryCrystallizationV1


@dataclass
class SourceResolutionResult:
    valid: bool
    errors: list[str] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)


async def resolve_memory_card_ref(pool: asyncpg.Pool, card_id: str) -> bool:
    row = await mc_dal.get_card(pool, card_id)
    return row is not None


async def resolve_grammar_event_ref(pool: asyncpg.Pool, event_id: str) -> bool:
    """Best-effort: grammar events may live in substrate tables or bus-only."""
    async with pool.acquire() as conn:
        try:
            row = await conn.fetchrow(
                """
                SELECT 1 FROM grammar_traces
                WHERE trace_id = $1 OR event_id = $1
                LIMIT 1
                """,
                event_id,
            )
            if row:
                return True
        except Exception:
            pass
        try:
            row = await conn.fetchrow(
                """
                SELECT 1 FROM substrate_grammar_events
                WHERE event_id = $1
                LIMIT 1
                """,
                event_id,
            )
            return row is not None
        except Exception:
            # Grammar may be bus-only; allow gev_ prefixed ids as soft-valid
            return str(event_id).startswith("gev_")


async def resolve_evidence_ref(pool: asyncpg.Pool, ref: CrystallizationEvidenceRefV1) -> bool:
    if ref.source_kind == "memory_card":
        return await resolve_memory_card_ref(pool, ref.source_id)
    if ref.source_kind == "grammar_event":
        return await resolve_grammar_event_ref(pool, ref.source_id)
    if ref.source_kind in ("operator_note", "chat_turn", "tool_result", "service_trace", "repo_event"):
        return bool((ref.source_id or "").strip())
    if ref.source_kind in ("rdf_memory_graph", "graphiti_episode"):
        return bool((ref.source_id or "").strip())
    return bool((ref.source_id or "").strip())


async def resolve_crystallization_sources(
    pool: asyncpg.Pool,
    crystallization: MemoryCrystallizationV1,
) -> SourceResolutionResult:
    errors: list[str] = []
    unresolved: list[str] = []

    for card_id in crystallization.source_card_ids:
        if not await resolve_memory_card_ref(pool, card_id):
            unresolved.append(f"memory_card:{card_id}")
            errors.append(f"unresolved memory_card source: {card_id}")

    for gev in crystallization.source_grammar_event_ids:
        if not await resolve_grammar_event_ref(pool, gev):
            unresolved.append(f"grammar_event:{gev}")
            errors.append(f"unresolved grammar_event source: {gev}")

    for ev in crystallization.evidence:
        if not await resolve_evidence_ref(pool, ev):
            unresolved.append(f"{ev.source_kind}:{ev.source_id}")
            errors.append(f"unresolved evidence: {ev.source_kind}:{ev.source_id}")

    return SourceResolutionResult(valid=not errors, errors=errors, unresolved=unresolved)
