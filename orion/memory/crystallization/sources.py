from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import asyncpg

from orion.core.storage import memory_cards as mc_dal
from orion.memory.crystallization.schemas import CrystallizationEvidenceRefV1, MemoryCrystallizationV1

logger = logging.getLogger("orion.memory.crystallization.sources")


@dataclass
class SourceResolutionResult:
    valid: bool
    errors: list[str] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)
    # Refs that pointed at a real grammar event which retention has since deleted. NOT an
    # error, and deliberately NOT folded into `unresolved` -- quarantining a proposal because
    # the substrate did its own housekeeping would be a false positive, and one that grows
    # every day. Reported so "the evidence aged out" stays distinguishable from "the evidence
    # was never there", which is the entire distinction this module exists to make.
    pruned: list[str] = field(default_factory=list)


async def resolve_memory_card_ref(pool: asyncpg.Pool, card_id: str) -> bool:
    row = await mc_dal.get_card(pool, card_id)
    return row is not None


async def grammar_retention_horizon(pool: asyncpg.Pool) -> Optional[datetime]:
    """Oldest grammar event still on disk, or None if unknown.

    This is the line between "this reference is broken" and "this reference is older than
    what the substrate keeps". `grammar_events` is bounded by GRAMMAR_EVENTS_RETENTION_DAYS
    (services/orion-sql-writer/app/grammar_truth.py), so a reference minted months ago SHOULD
    fail to resolve, and treating that as corruption would quarantine an ever-growing share
    of perfectly good crystallizations.

    Returns None on any failure. Callers must treat None as "cannot classify", never as
    "no retention" -- guessing in that direction is what turns an outage into mass
    quarantine.
    """
    try:
        async with pool.acquire() as conn:
            return await conn.fetchval("SELECT MIN(created_at) FROM grammar_events")
    except Exception:
        logger.warning("grammar_retention_horizon_unavailable", exc_info=True)
        return None


async def resolve_grammar_event_ref(pool: asyncpg.Pool, event_id: str) -> bool:
    """Does this grammar event id exist on disk right now?

    WHAT THIS REPLACES. The previous implementation queried
    `grammar_traces WHERE trace_id = $1 OR event_id = $1` -- but `grammar_traces` has no
    `event_id` column, so that statement raised on EVERY call and was swallowed by a bare
    `except Exception: pass`. It then fell through to `substrate_grammar_events`, a table
    that does not exist in this database, which raised too. The function's actual behaviour
    was therefore its last line and nothing else:

        return str(event_id).startswith("gev_")

    A string prefix check wearing two dead SQL queries as a costume. Confirmed live
    2026-08-20: all 1,167 distinct referenced ids are `gev_`-prefixed, so it returned True
    for every one of them, including the 876 whose events no longer exist and the 14 that
    never did.

    The real table is `grammar_events`, which has both `event_id` and `trace_id`. A grammar
    reference may legitimately name either -- a trace id identifies the whole reasoning
    episode, an event id one step within it -- so both are accepted.
    """
    try:
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT 1 FROM grammar_events
                WHERE event_id = $1 OR trace_id = $1
                LIMIT 1
                """,
                event_id,
            )
            return row is not None
    except Exception:
        # Deliberately NOT a silent False. An unreachable database is not evidence that a
        # reference is bad, and returning False here would quarantine every proposal
        # validated during an outage. Raise, and let the caller decide.
        logger.exception("grammar_event_ref_probe_failed event_id=%s", event_id)
        raise


async def resolve_evidence_ref(pool: asyncpg.Pool, ref: CrystallizationEvidenceRefV1) -> bool:
    if ref.source_kind == "memory_card":
        return await resolve_memory_card_ref(pool, ref.source_id)
    if ref.source_kind == "grammar_event":
        return await resolve_grammar_event_ref(pool, ref.source_id)
    if ref.source_kind in ("operator_note", "chat_turn", "tool_result", "service_trace", "repo_event", "autonomy_episode"):
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

    # Grammar refs are the only source kind whose store is BOUNDED, so "absent" has two very
    # different meanings and they must not be collapsed. The horizon is fetched once, not per
    # ref, and only if there is anything to classify.
    pruned: list[str] = []
    if crystallization.source_grammar_event_ids:
        horizon = await grammar_retention_horizon(pool)
        # Was this proposal minted before the oldest surviving grammar event? If so, every
        # event it referenced has aged out by construction, and a missing ref says nothing
        # about the proposal's quality. Verified against live data 2026-08-20: this rule
        # splits 876 aged-out refs from 14 genuinely-missing ones, where the previous code
        # passed all 1,167 without looking.
        created_at = getattr(crystallization, "created_at", None)
        predates_horizon = bool(
            horizon is not None
            and isinstance(created_at, datetime)
            and created_at < horizon
        )
        for gev in crystallization.source_grammar_event_ids:
            try:
                found = await resolve_grammar_event_ref(pool, gev)
            except Exception:
                # The probe could not run. Do not invent an answer in either direction --
                # False mass-quarantines during an outage, True silently re-creates the bug
                # this patch exists to fix. Record it as an explicit, visible error.
                unresolved.append(f"grammar_event:{gev}")
                errors.append(f"unresolvable grammar_event source (probe failed): {gev}")
                continue
            if found:
                continue
            if predates_horizon:
                pruned.append(f"grammar_event:{gev}")
                continue
            unresolved.append(f"grammar_event:{gev}")
            errors.append(f"unresolved grammar_event source: {gev}")

    for ev in crystallization.evidence:
        if not await resolve_evidence_ref(pool, ev):
            unresolved.append(f"{ev.source_kind}:{ev.source_id}")
            errors.append(f"unresolved evidence: {ev.source_kind}:{ev.source_id}")

    if pruned:
        logger.info(
            "crystallization_sources_pruned_refs crystallization_id=%s count=%s -- "
            "these named real grammar events that retention has since deleted; not an error",
            getattr(crystallization, "crystallization_id", None),
            len(pruned),
        )
    return SourceResolutionResult(
        valid=not errors, errors=errors, unresolved=unresolved, pruned=pruned
    )
