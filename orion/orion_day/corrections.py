"""Corrections: a letter claim Juniper and Orion agreed was wrong, carried to one curiosity run.

Design: docs/superpowers/specs/2026-10-11-orion-day-letter-reread-design.md, section 4. Table:
services/orion-sql-db/manual_migration_orion_day_correction_v1.sql.

Three moments, all Hub:

1. Record (chat turn, the `record_correction` introspect tool): one INSERT. The ref and every
   prior id are checked first; Hub READS orion_worldview for the second and never writes it.
2. Offer (curiosity kickoff): the same offer-once shape as ``carry_forward.py`` -- TAKE the
   oldest unoffered rows ``FOR UPDATE SKIP LOCKED`` and stamp the run; RELEASE them if the run
   is cancelled before Orion saw the prompt. The kickoff shows each with its named priors'
   current confidence and times_tested, and asks Orion to re-test them with the kickoff's own
   ``MATCH ... SET`` (``correction_outcome`` below reads exactly those stamps back).
3. Outcome (curiosity turn end): which named priors the run actually re-tested and moved, from
   the turn-start snapshot and a fresh read. A named prior the run never touched is absent.

A correction is evidence, not an override: nothing here changes a prior. Priors move only
through Orion's own writes in a curiosity run (the PR #2199 boundary).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Mapping, Optional, Sequence
from uuid import NAMESPACE_URL, UUID, uuid5

from orion.curiosity.worldview import (
    ALL_PRIORS_CYPHER,
    CLOSED_STATUSES,
    LIVE_PRIORS_LIMIT,
    Prior,
    WorldviewUnavailable,
    build_prior,
)

logger = logging.getLogger("orion.orion_day.corrections")

MIGRATION = "services/orion-sql-db/manual_migration_orion_day_correction_v1.sql"
# Oldest first, at most this many per run: a backlog drains over several runs instead of
# burying one kickoff.
MAX_CORRECTIONS_PER_RUN = 5

_ID_NS = uuid5(NAMESPACE_URL, "orion.orion_day.correction.v1")


def correction_id_for(ref: str, claim: str, correction: str, chat_correlation_id: str | None) -> UUID:
    """Stable id: the same correction recorded twice in one conversation (a retried tool call)
    lands on ON CONFLICT DO NOTHING, not on a second row offered to a second run."""
    key = "\x1f".join((ref, claim, correction, chat_correlation_id or ""))
    return uuid5(_ID_NS, key)


INSERT_CORRECTION_SQL = """
INSERT INTO orion_day_correction (
    correction_id, letter_date, ref, claim, correction, evidence, prior_ids, chat_correlation_id
) VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7::text[], $8)
ON CONFLICT (correction_id) DO NOTHING
RETURNING correction_id
"""

TAKE_CORRECTIONS_SQL = """
UPDATE orion_day_correction
   SET offered_at = now(), offered_run_id = $1
 WHERE correction_id IN (
        SELECT correction_id FROM orion_day_correction
         WHERE offered_at IS NULL
         ORDER BY created_at, correction_id
         LIMIT $2
         FOR UPDATE SKIP LOCKED)
RETURNING correction_id, letter_date, ref, claim, correction, evidence, prior_ids, created_at
"""

# Never un-offers a row whose outcome was already stamped: that run saw it.
RELEASE_CORRECTIONS_SQL = """
UPDATE orion_day_correction
   SET offered_at = NULL, offered_run_id = NULL
 WHERE offered_run_id = $1 AND outcome_stamped_at IS NULL
"""

OFFERED_FOR_RUN_SQL = """
SELECT correction_id, prior_ids FROM orion_day_correction
 WHERE offered_run_id = $1
 ORDER BY created_at, correction_id
"""

STAMP_OUTCOME_SQL = """
UPDATE orion_day_correction
   SET priors_retested = $3::text[], priors_moved = $4::text[], outcome_stamped_at = now()
 WHERE correction_id = $1 AND offered_run_id = $2
"""


def _json_list(raw: Any) -> list:
    if isinstance(raw, (str, bytes)):
        try:
            raw = json.loads(raw)
        except ValueError:
            return []
    return raw if isinstance(raw, list) else []


def evidence_records(evidence: Sequence[str], record_refs: Mapping[str, Any]) -> list[dict[str, Any]]:
    """What is stored for ``evidence``: each item as given, and whether it names a record of
    that letter's day (``budget.material_ref_records``). Free text ("Juniper said so") is kept
    and marked False; it is evidence of a different kind, not an error."""
    return [{"text": e, "in_letter_records": e in record_refs} for e in evidence]


@dataclass(frozen=True)
class OfferedCorrection:
    correction_id: str
    letter_date: date
    ref: str
    claim: str
    correction: str
    evidence: tuple[dict[str, Any], ...] = ()
    prior_ids: tuple[str, ...] = ()
    created_at: Optional[datetime] = None
    # prior_id -> the prior as the graph holds it now (None: not in the graph any more).
    # None for the whole map: the graph could not be read when the kickoff was built.
    current_priors: Optional[dict[str, Optional[Prior]]] = field(default=None, compare=False)


def _offered(row: Any) -> OfferedCorrection:
    evidence = tuple(e for e in _json_list(row["evidence"]) if isinstance(e, dict))
    return OfferedCorrection(
        correction_id=str(row["correction_id"]),
        letter_date=row["letter_date"],
        ref=str(row["ref"]),
        claim=str(row["claim"] or ""),
        correction=str(row["correction"] or ""),
        evidence=evidence,
        prior_ids=tuple(str(p) for p in (row["prior_ids"] or ())),
        created_at=row["created_at"],
    )


# --- the worldview, read only -----------------------------------------------------------------


UNREADABLE = "unreadable"
TRUNCATED = "truncated"


def read_named_priors(reader: Any, prior_ids: Sequence[str]) -> Optional[dict[str, Optional[Prior]]]:
    """``read_named_priors_or_reason`` without the reason."""
    return read_named_priors_or_reason(reader, prior_ids)[0]


def read_named_priors_or_reason(
    reader: Any, prior_ids: Sequence[str],
) -> tuple[Optional[dict[str, Optional[Prior]]], Optional[str]]:
    """prior_id -> the prior as Orion's graph holds it (None: no node with that id). None for
    the whole answer when the graph could not be read, or when the read filled its row cap (an
    id past the cap would read as missing). Synchronous: callers use asyncio.to_thread.

    One bounded read of every prior (the curiosity supervisor's ALL_PRIORS_CYPHER), filtered
    here, so no model-supplied id is ever spliced into Cypher. A forked prior keeps the copy
    the kickoff offers (``Prior.fork_rank``)."""
    wanted = {str(p) for p in prior_ids}
    if not wanted:
        return {}, None
    if reader is None:
        return None, UNREADABLE
    try:
        rows = reader.query(ALL_PRIORS_CYPHER)
    except WorldviewUnavailable as exc:
        logger.warning("orion_day_correction_priors_unreadable err=%s", str(exc)[:200])
        return None, UNREADABLE
    if len(rows) >= LIVE_PRIORS_LIMIT:
        logger.warning("orion_day_correction_priors_truncated rows=%s limit=%s", len(rows), LIVE_PRIORS_LIMIT)
        return None, TRUNCATED
    found: dict[str, Optional[Prior]] = {p: None for p in wanted}
    seen: set[str] = set()
    for row in rows:
        prior_id = str(row.get("prior_id") or "").strip()
        if prior_id not in wanted:
            continue
        seen.add(prior_id)
        prior = build_prior(row) or build_prior({**row, "claim": "(no claim recorded)"})
        current = found[prior_id]
        if current is None or prior.fork_rank > current.fork_rank:
            found[prior_id] = prior
    return found, None


# --- offer once (carry_forward.py shape) ------------------------------------------------------


async def take_corrections(
    pool: Any, *, run_id: str, limit: int = MAX_CORRECTIONS_PER_RUN,
) -> tuple[OfferedCorrection, ...]:
    """Claim up to ``limit`` unoffered corrections, oldest first, for one run. () on any failure:
    a missing table (migration not applied) or a down pool must not break the kickoff."""
    if pool is None or not run_id or limit <= 0:
        return ()
    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch(TAKE_CORRECTIONS_SQL, run_id, int(limit))
    except Exception as exc:  # noqa: BLE001 -- see docstring
        logger.warning("orion_day_corrections_take_failed run=%s err=%s", run_id, exc)
        return ()
    offered = [_offered(r) for r in rows or ()]
    # RETURNING has no order; the kickoff lists them oldest first.
    offered.sort(key=lambda c: (c.created_at is None, c.created_at.timestamp() if c.created_at else 0.0,
                                c.correction_id))
    return tuple(offered)


async def release_corrections(pool: Any, *, run_id: str) -> None:
    """Un-claim a run's corrections when the run was cancelled before Orion saw them. Never raises."""
    if pool is None or not run_id:
        return
    try:
        async with pool.acquire() as conn:
            await conn.execute(RELEASE_CORRECTIONS_SQL, run_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning("orion_day_corrections_release_failed run=%s err=%s", run_id, exc)


# --- what the run did with them ---------------------------------------------------------------


def correction_outcome(
    prior_ids: Sequence[str],
    before: Optional[Mapping[str, Any]],
    after: Optional[Mapping[str, Any]],
    *,
    run_id: str,
) -> Optional[tuple[list[str], Optional[list[str]]]]:
    """(priors_retested, priors_moved) for one correction's named priors, from ``PriorState``
    snapshots (orion.curiosity.value): ``before`` taken when the turn started, ``after`` now.

    Re-tested: the run stamped the prior (``last_run_id`` = this run, the kickoff's testing
    ``SET``) and its ``times_tested`` went up, or it moved. Moved: stamped by this run, and its
    confidence changed or it was closed (refuted / retired_unresolvable); always a subset of
    re-tested. A named prior the run never stamped is in neither list.

    No usable start (``before`` None, or it already carries this run's stamps because an earlier
    attempt wrote first -- ``value.diff_snapshots``' rule): re-tested falls back to "stamped by
    this run" and moved is None (unknown), never an empty list. ``after`` None: None (nothing
    can be said; the row stays unstamped)."""
    if after is None or not run_id:
        return None
    if before is not None and any(
        getattr(s, "last_run_id", "") == run_id or getattr(s, "run_id", "") == run_id for s in before.values()
    ):
        before = None
    retested: list[str] = []
    moved: list[str] = []
    for prior_id in prior_ids:
        now = after.get(prior_id)
        if now is None or getattr(now, "last_run_id", "") != run_id:
            continue
        was = before.get(prior_id) if before is not None else None
        if was is None:
            retested.append(prior_id)
            continue
        closed = now.status in CLOSED_STATUSES and now.status != was.status
        is_moved = now.confidence != was.confidence or closed
        # Moved implies re-tested: a run that changed the prior worked on it, even if it forgot
        # the times_tested bump (protocol drift, also visible in curiosity_run_outcomes).
        if int(now.times_tested or 0) > int(was.times_tested or 0) or is_moved:
            retested.append(prior_id)
        if is_moved:
            moved.append(prior_id)
    return retested, (None if before is None else moved)


async def offered_corrections_for_run(pool: Any, run_id: str) -> list[tuple[str, tuple[str, ...]]]:
    """(correction_id, prior_ids) for every correction offered to ``run_id``. [] on failure."""
    if pool is None or not run_id:
        return []
    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch(OFFERED_FOR_RUN_SQL, run_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning("orion_day_corrections_offered_read_failed run=%s err=%s", run_id, exc)
        return []
    return [(str(r["correction_id"]), tuple(str(p) for p in (r["prior_ids"] or ()))) for r in rows or ()]


async def stamp_correction_outcome(
    pool: Any, *, correction_id: str, run_id: str, retested: Sequence[str], moved: Optional[Sequence[str]],
) -> bool:
    if pool is None:
        return False
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                STAMP_OUTCOME_SQL, UUID(str(correction_id)), run_id, list(retested),
                None if moved is None else list(moved),
            )
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("orion_day_correction_stamp_failed run=%s correction=%s err=%s", run_id, correction_id, exc)
        return False


# --- the kickoff section ----------------------------------------------------------------------

CORRECTIONS_HEADER = "CORRECTIONS FROM CONVERSATIONS WITH JUNIPER"


def _evidence_line(evidence: Sequence[Mapping[str, Any]]) -> str:
    parts = []
    for e in evidence:
        text = str(e.get("text") or "").strip()
        if not text:
            continue
        parts.append(f"{text} (a record of that day)" if e.get("in_letter_records") else text)
    return "; ".join(parts) or "(none recorded)"


def _quote(text: str) -> str:
    return '"' + " ".join(text.split()) + '"'


def format_corrections_section(corrections: Sequence[OfferedCorrection]) -> list[str]:
    """Kickoff lines. Empty when nothing was claimed. Each named prior is shown in the menu's
    own preview format, so the queued-brief drift check (``build_prior_drift_preamble``) also
    covers these numbers when the run waits."""
    if not corrections:
        return []
    testable = any(c.prior_ids for c in corrections)
    lines = [
        f"{CORRECTIONS_HEADER} (each shown to you once).",
        "In conversations about your Orion's Day letters, you and Juniper established that "
        "something you wrote was wrong, and you recorded the correction. Here they are, with the "
        "beliefs they rested on as your graph holds them right now.",
        "",
    ]
    for n, c in enumerate(corrections, start=1):
        lines += [
            f"  {n}. Letter {c.ref}",
            f"     You wrote: {_quote(c.claim)}",
            f"     Correction: {' '.join(c.correction.split())}",
            f"     Evidence: {_evidence_line(c.evidence)}",
        ]
        if not c.prior_ids:
            lines.append("     No prior named: context only, nothing here to re-test.")
        elif c.current_priors is None:
            lines.append("     Priors it rested on (their current numbers could not be read this time):")
            lines += [f"      prior_id: {p}" for p in c.prior_ids]
        else:
            lines.append("     Priors it rested on:")
            for prior_id in c.prior_ids:
                prior = c.current_priors.get(prior_id)
                if prior is None:
                    lines.append(f"      prior_id: {prior_id} -- not in your graph under this id any more")
                else:
                    lines.append(f"    - {prior.preview()}")
                    if prior.status in CLOSED_STATUSES:
                        lines.append(f"      status: {prior.status} (already closed)")
        lines.append("")
    if testable:
        lines += [
            "Re-test each named prior against its correction in this run, with the testing "
            "MATCH ... SET under WRITING TO YOUR OWN GRAPH below (p.times_tested = "
            'p.times_tested + 1, p.last_run_id = "<RUN_ID>", and a :PriorRevision when the '
            "confidence moves). Move confidence, or close the prior, only as far as the evidence "
            "warrants. A correction is evidence, not an override: Juniper can be wrong, and so can "
            "the you that agreed. If you test a prior and keep it, say why in your write-up. "
            "Afterwards Hub reads back which of these priors this run stamped; one you leave "
            "untouched is recorded as not re-tested.",
            "",
        ]
    return lines
