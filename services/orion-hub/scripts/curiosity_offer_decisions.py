"""The curiosity spend log: what each investigation run was offered, and what
it bought.

P1 phase 1 of docs/superpowers/specs/2026-09-25-attention-with-stakes-design.md.
Tables: `services/orion-sql-db/manual_migration_curiosity_spend_v1.sql`.

Three writes per investigation run, all by Hub:

1. At dispatch, `curiosity_offer_decisions`: the arm (value order vs today's
   uncertainty order), its propensity, and every prior offered with the
   expected value it was offered at. This is the choice set nothing recorded
   before.
2. When the turn starts, the same row's `turn_snapshot`: every prior's
   confidence, tested count and run stamps. Taken at turn START, not at
   dispatch, because a durable run can wait in admission for hours while
   other turns move priors. Only the first attempt writes it, so a retried
   run is scored against where it actually began.
3. When the turn ends, `curiosity_run_outcomes`: the diff, scored in nats
   (`orion/curiosity/value.py`), plus agreement with the `:PriorRevision`
   nodes Orion wrote by hand.

Best-effort throughout: every function logs and returns a neutral value
rather than raise into the curiosity loop. A missing table (migration not
applied) warns once per process, then stays quiet.

Reads the graph with the Atlas's existing prior query, which already returns
`run_id` and `last_run_id` -- no new Cypher.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Mapping, Optional, Sequence

from orion.curiosity.atlas import (
    ATLAS_PRIORS_CYPHER,
    LABEL_PRIOR_REVISION,
    run_nodes_cypher,
    valid_run_id,
)
from orion.curiosity.value import (
    PriorState,
    PriorTestRecord,
    RunOutcome,
    YieldModel,
    entropy_nats,
    index_states,
    prior_tests_from_rows,
    valid_confidence,
)
from orion.curiosity.worldview import Prior, WorldviewReader, WorldviewUnavailable

logger = logging.getLogger("orion-hub.curiosity_offer_decisions")

MIGRATION = "services/orion-sql-db/manual_migration_curiosity_spend_v1.sql"

INSERT_DECISION_SQL = """
INSERT INTO curiosity_offer_decisions (
    run_id, arm, value_arm_propensity, offered, stale_offered, material_ids, constants
) VALUES ($1, $2, $3, $4::jsonb, $5::jsonb, $6::jsonb, $7::jsonb)
ON CONFLICT (run_id) DO NOTHING
"""

SET_TURN_SNAPSHOT_SQL = """
UPDATE curiosity_offer_decisions
SET turn_snapshot = $2::jsonb, turn_started_at = now()
WHERE run_id = $1 AND turn_snapshot IS NULL
"""

LOAD_TURN_SNAPSHOT_SQL = """
SELECT turn_snapshot FROM curiosity_offer_decisions WHERE run_id = $1
"""

UPSERT_OUTCOME_SQL = """
INSERT INTO curiosity_run_outcomes (
    run_id, realized_nats, n_tested, n_moved, n_formed, n_moved_untested,
    n_unattributed, n_invalid_confidence, per_prior, revision_agreement
) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9::jsonb, $10)
ON CONFLICT (run_id) DO UPDATE SET
    completed_at = now(),
    realized_nats = EXCLUDED.realized_nats,
    n_tested = EXCLUDED.n_tested,
    n_moved = EXCLUDED.n_moved,
    n_formed = EXCLUDED.n_formed,
    n_moved_untested = EXCLUDED.n_moved_untested,
    n_unattributed = EXCLUDED.n_unattributed,
    n_invalid_confidence = EXCLUDED.n_invalid_confidence,
    per_prior = EXCLUDED.per_prior,
    revision_agreement = EXCLUDED.revision_agreement
"""

LOAD_HISTORY_SQL = """
SELECT per_prior FROM curiosity_run_outcomes
WHERE completed_at >= now() - ($1 * interval '1 day')
ORDER BY completed_at ASC
"""

_warned_missing_table = False


def _log_failure(what: str, run_id: Optional[str], exc: BaseException) -> None:
    global _warned_missing_table
    text = str(exc)
    if "does not exist" in text:
        if not _warned_missing_table:
            _warned_missing_table = True
            logger.warning(
                "curiosity_spend_log_table_missing op=%s -- apply %s; until then "
                "offers and outcomes are not recorded and value ordering runs on a "
                "cold (uncertainty-equivalent) model",
                what,
                MIGRATION,
            )
        return
    logger.warning("curiosity_spend_log_failed op=%s run=%s err=%s", what, run_id, text[:300])


def _json_list(raw: Any) -> list:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            return []
    return raw if isinstance(raw, list) else []


# --- graph reads (sync; call through asyncio.to_thread) ---------------------


def read_prior_states(reader: WorldviewReader) -> Optional[dict[str, PriorState]]:
    """Every prior, live and closed, as a snapshot. None when the graph could
    not answer -- an unreadable graph is not an empty one."""
    try:
        rows = reader.query(ATLAS_PRIORS_CYPHER)
    except WorldviewUnavailable as exc:
        logger.warning("curiosity_spend_snapshot_unreadable err=%s", str(exc)[:200])
        return None
    return index_states(s for s in (PriorState.from_json(r) for r in rows) if s is not None)


def read_run_revisions(
    reader: WorldviewReader, run_id: str
) -> Optional[dict[str, tuple[Optional[float], Optional[float]]]]:
    """prior_id -> (from, to) for the `:PriorRevision` nodes this run wrote.
    A prior revised twice in one run keeps its first `from` and last `to`."""
    rid = valid_run_id(run_id)
    if rid is None:
        return None
    try:
        rows = reader.query(run_nodes_cypher(LABEL_PRIOR_REVISION, [rid]))
    except WorldviewUnavailable as exc:
        logger.warning("curiosity_spend_revisions_unreadable run=%s err=%s", rid, str(exc)[:200])
        return None
    rows = sorted(rows, key=lambda r: str(r.get("written_at") or ""))
    out: dict[str, tuple[Optional[float], Optional[float]]] = {}
    for row in rows:
        prior_id = str(row.get("prior_id") or "").strip()
        if not prior_id:
            continue
        first_from = out[prior_id][0] if prior_id in out else valid_confidence(row.get("from_confidence"))
        out[prior_id] = (first_from, valid_confidence(row.get("to_confidence")))
    return out


# --- what to record ----------------------------------------------------------


def offered_rows(priors: Sequence[Prior], model: YieldModel) -> list[dict[str, Any]]:
    """Each offered prior, in the order Orion was shown it, with the numbers
    it was offered at."""
    rows = []
    for rank, prior in enumerate(priors, start=1):
        rows.append(
            {
                "rank": rank,
                "prior_id": prior.prior_id,
                "confidence": prior.confidence,
                "times_tested": prior.times_tested,
                "entropy_nats": entropy_nats(prior.confidence),
                "yield": model.yield_for(prior.prior_id),
                "expected_nats": model.expected_nats(prior.prior_id, prior.confidence),
            }
        )
    return rows


# --- Postgres (asyncpg pool; best-effort) ------------------------------------


async def record_offer_decision(
    pool: Any,
    *,
    run_id: str,
    arm: str,
    value_arm_propensity: float,
    offered: list[dict[str, Any]],
    stale_offered: list[dict[str, Any]],
    material_ids: list[str],
    constants: Mapping[str, Any],
) -> bool:
    if pool is None:
        return False
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                INSERT_DECISION_SQL,
                run_id,
                arm,
                float(value_arm_propensity),
                json.dumps(offered),
                json.dumps(stale_offered),
                json.dumps(material_ids),
                json.dumps(dict(constants)),
            )
        return True
    except Exception as exc:  # noqa: BLE001
        _log_failure("record_offer_decision", run_id, exc)
        return False


async def record_turn_snapshot(
    pool: Any, run_id: str, states: Mapping[str, PriorState]
) -> bool:
    """First attempt wins (`turn_snapshot IS NULL`). A run with no decision
    row -- not an investigation, or its dispatch write failed -- is a no-op."""
    if pool is None:
        return False
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                SET_TURN_SNAPSHOT_SQL,
                run_id,
                json.dumps([s.as_json() for s in states.values()]),
            )
        return True
    except Exception as exc:  # noqa: BLE001
        _log_failure("record_turn_snapshot", run_id, exc)
        return False


async def load_turn_snapshot(
    pool: Any, run_id: str
) -> tuple[bool, Optional[dict[str, PriorState]]]:
    """(decision row found, start snapshot). Found with no snapshot means the
    graph was unreadable when the turn began: the run is scored as unknown.
    Not found means there is nothing to score (not a dispatched investigation,
    or the log could not be read) and the caller records nothing."""
    if pool is None:
        return False, None
    try:
        async with pool.acquire() as conn:
            row = await conn.fetchrow(LOAD_TURN_SNAPSHOT_SQL, run_id)
    except Exception as exc:  # noqa: BLE001
        _log_failure("load_turn_snapshot", run_id, exc)
        return False, None
    if row is None:
        return False, None
    if row["turn_snapshot"] is None:
        return True, None
    states = (PriorState.from_json(r) for r in _json_list(row["turn_snapshot"]) if isinstance(r, dict))
    return True, index_states(s for s in states if s is not None)


async def record_run_outcome(
    pool: Any, run_id: str, outcome: RunOutcome, agreement: Optional[float]
) -> bool:
    if pool is None:
        return False
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                UPSERT_OUTCOME_SQL,
                run_id,
                outcome.realized_nats,
                outcome.n_tested,
                outcome.n_moved,
                outcome.n_formed,
                outcome.n_moved_untested,
                outcome.n_unattributed,
                outcome.n_invalid_confidence,
                json.dumps([o.as_json() for o in outcome.per_prior]),
                agreement,
            )
        return True
    except Exception as exc:  # noqa: BLE001
        _log_failure("record_run_outcome", run_id, exc)
        return False


async def load_prior_test_history(pool: Any, *, days: float) -> list[PriorTestRecord]:
    """Every scored test in the window, oldest first. [] when unavailable --
    the caller then runs on the cold model, which orders exactly as today."""
    if pool is None:
        return []
    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch(LOAD_HISTORY_SQL, float(days))
    except Exception as exc:  # noqa: BLE001
        _log_failure("load_prior_test_history", None, exc)
        return []
    history: list[PriorTestRecord] = []
    for row in rows:
        history.extend(prior_tests_from_rows(r for r in _json_list(row["per_prior"]) if isinstance(r, dict)))
    return history
