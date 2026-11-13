"""The personality reducer's reads (spec rev 4: choices only).

Spec: docs/superpowers/specs/2026-10-16-orion-dynamic-personality-graph-design.md (rev 4).
Writer: ``app.personality_driver``. Pure graph: ``orion.personality.graph``.

Two reads, both keyset cursors in ``personality_cursor``:

1. ``curiosity_sync`` -- curiosity runs (``curiosity_run_outcomes`` JOIN ``curiosity_offer_decisions``,
   read-only) in ``(completed_at, run_id)`` order. Each run becomes its ``PersonalityChoiceV1`` rows
   (``orion.personality.choices``), which the driver inserts into ``personality_choice`` (insert-only,
   ``ON CONFLICT DO NOTHING``). With no cursor yet the read starts at the epoch: that IS the one-time
   backfill of every curiosity choice already stored, and it is idempotent (``choice_id`` is
   deterministic per run and prior; ``completed_at`` is reset on every outcome upsert, so a run can be
   read twice and its rows are the same rows).
2. ``choice`` -- ``personality_choice`` rows in ``(created_at, choice_id)`` order: every lane (bus
   producers and the curiosity sync alike). These are the ONLY evidence the graph folds.

Both reads stop at ``hi`` (now minus the read lag, so a late-committing writer is not skipped) and
take at most ``PER_TICK_CAP`` rows a tick, so a backlog drains over several ticks.

Rev 2's prose sources (dream, journal, reverie, reading text, day letter, chat and outreach text,
self-inquiry seeds) are deleted. Their cursors stay in the table, unread.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Optional

from orion.personality.choices import curiosity_choices_from_rows
from orion.schemas.personality import PersonalityChoiceV1

STATEMENT_TIMEOUT = "30s"  # a stalled read fails the tick (retried next tick), never hangs it
READ_LAG_SEC = 120.0       # rows newer than now - lag wait for the next tick (late writers)
PER_TICK_CAP = 200         # rows per read per tick
CURIOSITY_CURSOR = "curiosity_sync"
CHOICE_CURSOR = "choice"
EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


@dataclass(frozen=True)
class Cursor:
    at: datetime
    ref: str = ""


def utc(value: Any) -> Optional[datetime]:
    if not isinstance(value, datetime):
        return None
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


# Keyset: (t, ref) > (cursor.at, cursor.ref) AND t < hi, written so the time index applies.
# personality_choice.created_at is the inserting transaction's start (DEFAULT now()); every insert
# is a short transaction, and READ_LAG_SEC (120 s) covers a late commit and DB/app clock skew.
def _keyset(t: str, ref: str) -> str:
    return (f"{t} >= %(at)s AND ({t} > %(at)s OR {ref} > %(ref)s) AND {t} < %(hi)s "
            f"ORDER BY {t}, {ref} LIMIT %(limit)s")


CURIOSITY_SQL = ("SELECT d.run_id, d.offered, d.stale_offered, d.turn_started_at, d.decided_at, d.arm, "
                 "o.per_prior, o.completed_at, o.turn_ok "
                 "FROM curiosity_run_outcomes o JOIN curiosity_offer_decisions d USING (run_id) "
                 # A self-inquiry run's decision row offers nothing: no choice to read (choices.py skips it too).
                 "WHERE d.arm <> 'self_inquiry' "
                 # A run is synced ONCE. Hub re-upserts an outcome (completed_at = now(), per_prior
                 # replaced), so a re-read run could carry a different tested set: new rows with a new
                 # picks_in_decision next to the old ones would weigh one decision twice.
                 "AND NOT EXISTS (SELECT 1 FROM personality_choice c WHERE c.act_id = 'curiosity_run:' || o.run_id) "
                 "AND " + _keyset("o.completed_at", "o.run_id"))

CHOICE_COLUMNS = ("choice_id, lane, occurred_at, chosen, alternatives, chosen_market_rank, declined_all, "
                  "care_promoted, picks_in_decision, situation, act_id, correlation_id, written_by, created_at")
CHOICE_SQL = f"SELECT {CHOICE_COLUMNS} FROM personality_choice WHERE " + _keyset("created_at", "choice_id")

_OFFER_KEYS = ("run_id", "offered", "stale_offered", "turn_started_at", "decided_at", "arm")
_OUTCOME_KEYS = ("run_id", "per_prior", "completed_at", "turn_ok")

# Self-naming answers (orion_disposition_naming, written by orion-sql-writer). Never folded as
# evidence; ``graph.apply_namings`` applies them after the fold, in the same tick.
NAMING_SQL = ("SELECT naming_id, disposition_id, verdict, hub_check, name, sentence, merge_with, written_at, "
              "created_at FROM orion_disposition_naming WHERE " + _keyset("created_at", "naming_id"))
NAMING_EPOCH = EPOCH  # no cursor yet: read every answer ever written


def curiosity_run_choices(r: Mapping[str, Any]) -> list[PersonalityChoiceV1]:
    """One joined curiosity row -> its choices (possibly none)."""
    return curiosity_choices_from_rows({k: r.get(k) for k in _OFFER_KEYS}, {k: r.get(k) for k in _OUTCOME_KEYS})


def _j(value: Any) -> Any:
    return json.loads(value) if isinstance(value, (str, bytes)) else value


def choice_from_row(r: Mapping[str, Any]) -> Optional[PersonalityChoiceV1]:
    """A ``personality_choice`` row back into the contract, or None when it no longer validates
    (the driver consumes it as skipped evidence)."""
    try:
        return PersonalityChoiceV1(
            choice_id=r["choice_id"], lane=r["lane"], occurred_at=utc(r["occurred_at"]),
            chosen=_j(r.get("chosen")), alternatives=_j(r.get("alternatives")) or [],
            chosen_market_rank=r.get("chosen_market_rank"), declined_all=bool(r.get("declined_all")),
            care_promoted=bool(r.get("care_promoted")), picks_in_decision=int(r.get("picks_in_decision") or 1),
            situation=_j(r.get("situation")) or {}, act_id=r.get("act_id"), correlation_id=r.get("correlation_id"),
        )
    except Exception:  # noqa: BLE001 - a bad stored row is skipped, never a failed tick
        return None


async def _fetch(conn: Any, sql: str, params: dict) -> list[dict]:
    cur = await conn.execute(sql, params)
    return list(await cur.fetchall())


class PersonalitySourceReader:
    """Read-only transactions on the service's psycopg pool (dict rows)."""

    def __init__(self, pool: Any, *, limit: int = PER_TICK_CAP) -> None:
        self._pool = pool
        self._limit = limit

    async def _read(self, sql: str, cursor: Cursor, hi: datetime) -> list[dict]:
        if cursor.at >= hi:
            return []
        async with self._pool.connection() as conn:
            async with conn.transaction():
                await conn.execute("SET TRANSACTION READ ONLY")
                await conn.execute(f"SET LOCAL statement_timeout = '{STATEMENT_TIMEOUT}'")
                rows = await _fetch(conn, sql, {"at": cursor.at, "ref": cursor.ref, "hi": hi, "limit": self._limit})
        return [dict(r) for r in rows]

    async def read_curiosity(self, cursor: Optional[Cursor], hi: datetime) -> list[dict]:
        """Joined curiosity runs after ``cursor`` (epoch when None: the backfill) and before ``hi``."""
        return await self._read(CURIOSITY_SQL, cursor or Cursor(EPOCH, ""), hi)

    async def read_choices(self, cursor: Optional[Cursor], hi: datetime) -> list[dict]:
        """``personality_choice`` rows after ``cursor`` (epoch when None) and before ``hi``."""
        return await self._read(CHOICE_SQL, cursor or Cursor(EPOCH, ""), hi)

    async def read_namings(self, cursor: Optional[Cursor], hi: datetime) -> list[dict]:
        """Naming rows strictly after ``cursor`` and before ``hi``. With no cursor, from the start."""
        return await self._read(NAMING_SQL, cursor or Cursor(NAMING_EPOCH, ""), hi)
