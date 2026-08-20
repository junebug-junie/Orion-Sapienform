from __future__ import annotations

import logging
from dataclasses import dataclass, field

from typing import Optional  # noqa: F401 - re-exported for callers

import asyncpg

from orion.core.storage import memory_cards as mc_dal
from orion.memory.crystallization.schemas import CrystallizationEvidenceRefV1, MemoryCrystallizationV1

logger = logging.getLogger("orion.memory.crystallization.sources")


@dataclass
class SourceResolutionResult:
    valid: bool
    errors: list[str] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)
    # Grammar refs that do not resolve against `grammar_events` right now. Deliberately NOT
    # called "pruned": we genuinely cannot tell an aged-out reference from one that never
    # existed, and an earlier version of this file claimed it could. Reported, never fatal --
    # see the long note on _classify_grammar_refs for why absence here must not invalidate.
    absent_grammar_refs: list[str] = field(default_factory=list)
    # Refs whose existence could not be checked at all (database unreachable mid-validation).
    # Distinct from `absent`: "I looked and it is gone" and "I could not look" are different
    # facts, and collapsing them is how an outage turns into mass quarantine.
    unverified_grammar_refs: list[str] = field(default_factory=list)


async def resolve_memory_card_ref(pool: asyncpg.Pool, card_id: str) -> bool:
    row = await mc_dal.get_card(pool, card_id)
    return row is not None


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


# WHY AN ABSENT GRAMMAR REF IS NEVER FATAL.
#
# `grammar_events` is the only source store with RETENTION (3 days,
# GRAMMAR_EVENTS_RETENTION_DAYS). Every grammar reference is therefore perishable by
# construction: given enough time, every crystallization's grammar evidence stops resolving.
# Treating that as a defect quarantines an ever-growing share of perfectly good proposals.
#
# THE INFERENCE THAT DOES NOT WORK, AND WHY IT IS NOT WORTH RETRYING. The first version of
# this classified an absent ref as "pruned" (benign) when the crystallization's own
# created_at predated the live retention horizon, and "missing" (fatal) otherwise. Code
# review killed it with live data:
#
#   * Crystallizations COPY REFS FORWARD. `65b0662d` inherited seven ids verbatim from
#     `4b4bd619`, minted 25 hours earlier. The same seven ids got opposite verdicts from the
#     two carriers -- proof the rule measured the carrier, not the reference.
#   * Refs are not contemporaneous with their carrier at all: p50 lag 495s, p95 18.3 HOURS,
#     max 43.6 hours. Against a 3-day window that is a large fraction of the whole budget.
#   * Every one of the 14 refs the rule called "genuinely missing" was in fact aged-out. A
#     100% false-positive rate on its own error bucket.
#
# Nor does the ref row's own `memory_crystallization_sources.created_at` help: it defaults to
# now() at INSERT, so for a copied ref it carries the COPYING proposal's timestamp, not the
# original event's. Checked before relying on it.
#
# So there is no timestamp on either side that distinguishes "aged out" from "never existed",
# and a validator that claims to make that distinction is confabulating. The honest contract:
# LOOK IT UP FOR REAL, REPORT WHAT WAS FOUND, and never invalidate a proposal because the
# substrate did its own housekeeping. If the distinction is ever genuinely needed, the sound
# fix is to persist the resolution outcome when the ref is first recorded, not to re-derive
# it from clocks afterwards.
def _grammar_ref_ids(crystallization: MemoryCrystallizationV1) -> list[str]:
    """Every grammar id this proposal names, from BOTH carriers, de-duplicated, in order.

    `source_grammar_event_ids` and `evidence[kind=grammar_event]` overlap heavily -- live
    2026-08-20, all 61 affected crystallizations carried the same non-resolving ids in both.
    The first version of this walked them separately, so refs excused by one loop were
    re-flagged as fatal by the other and 61 of 61 proposals still quarantined. The feature
    was inert in production while 12 tests passed, because every test fixture had an empty
    evidence list.
    """
    seen: set[str] = set()
    out: list[str] = []
    for gev in crystallization.source_grammar_event_ids:
        if gev and gev not in seen:
            seen.add(gev)
            out.append(gev)
    for ev in crystallization.evidence:
        if ev.source_kind != "grammar_event":
            continue
        sid = (ev.source_id or "").strip()
        if sid and sid not in seen:
            seen.add(sid)
            out.append(sid)
    return out


async def resolve_crystallization_sources(
    pool: asyncpg.Pool,
    crystallization: MemoryCrystallizationV1,
) -> SourceResolutionResult:
    errors: list[str] = []
    unresolved: list[str] = []
    absent: list[str] = []
    unverified: list[str] = []

    for card_id in crystallization.source_card_ids:
        if not await resolve_memory_card_ref(pool, card_id):
            unresolved.append(f"memory_card:{card_id}")
            errors.append(f"unresolved memory_card source: {card_id}")

    # One pass over the union of both carriers, so a ref cannot be excused here and condemned
    # in the evidence loop below.
    for gev in _grammar_ref_ids(crystallization):
        try:
            found = await resolve_grammar_event_ref(pool, gev)
        except Exception:
            # "I could not look" is not "it is gone". Recorded and surfaced, but NOT fatal:
            # making it fatal quarantines every proposal validated during a database blip,
            # and persists that quarantine to disk.
            logger.warning("grammar_event_ref_unverified event_id=%s", gev)
            unverified.append(f"grammar_event:{gev}")
            continue
        if not found:
            absent.append(f"grammar_event:{gev}")

    # Non-grammar evidence still resolves normally and still invalidates: those stores are
    # unbounded, so absence there really does mean the reference is broken.
    for ev in crystallization.evidence:
        if ev.source_kind == "grammar_event":
            continue  # handled above, against the deduplicated union
        if not await resolve_evidence_ref(pool, ev):
            unresolved.append(f"{ev.source_kind}:{ev.source_id}")
            errors.append(f"unresolved evidence: {ev.source_kind}:{ev.source_id}")

    if absent or unverified:
        logger.info(
            "crystallization_grammar_refs crystallization_id=%s absent=%s unverified=%s "
            "-- grammar_events is retention-bounded, so absence is expected and not an error",
            getattr(crystallization, "crystallization_id", None),
            len(absent),
            len(unverified),
        )
    return SourceResolutionResult(
        valid=not errors,
        errors=errors,
        unresolved=unresolved,
        absent_grammar_refs=absent,
        unverified_grammar_refs=unverified,
    )
