"""The curiosity spend log: what each investigation run was offered, and what
it bought.

P1 phase 1 of docs/superpowers/specs/2026-09-25-attention-with-stakes-design.md.
Tables: `services/orion-sql-db/manual_migration_curiosity_spend_v1.sql`.

Three writes per investigation run, all by Hub:

1. At dispatch, `curiosity_offer_decisions`: the arm (value order vs today's
   uncertainty order), its propensity, and every prior offered with the
   expected value it was offered at. This is the choice set nothing recorded
   before.
2. When the turn starts, the same row's `turn_started_at` and
   `turn_snapshot`: every prior's confidence, tested count and run stamps.
   Taken at turn START, not at dispatch, because a durable run can wait in
   admission for hours while other turns move priors. The first snapshot
   taken wins, so a retried run is scored from where it began. If the first
   attempt could not read the graph, a retry may take the snapshot -- and the
   score refuses any start that already carries this run's own stamps
   (`value.diff_snapshots`), so a retry is scored only when the attempt
   before it wrote nothing: from where the run really began, or not at all.
3. When the turn ends, `curiosity_run_outcomes`: the diff, scored in nats
   (`orion/curiosity/value.py`), whether the turn produced text (`turn_ok`),
   why the score is unknown when it is (`unknown_reason`), plus agreement
   with the `:PriorRevision` nodes Orion wrote by hand.

Best-effort throughout: every function logs and returns a neutral value
rather than raise into the curiosity loop. A missing table (migration not
applied) or a missing pool warns once per process, then stays quiet; any
other database error warns every time.

Reads the graph with the Atlas's existing prior query, which already returns
`run_id` and `last_run_id` -- no new Cypher. That query is capped at
`ATLAS_PRIORS_LIMIT` rows with no order, so a snapshot that fills the cap may
be missing priors and is treated as unreadable.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Mapping, Optional, Sequence

from orion.curiosity.atlas import (
    ATLAS_PRIORS_CYPHER,
    ATLAS_PRIORS_LIMIT,
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
from orion.curiosity.worldview import Prior, WorldviewReader, WorldviewUnavailable, build_prior

logger = logging.getLogger("orion-hub.curiosity_offer_decisions")

MIGRATION = "services/orion-sql-db/manual_migration_curiosity_spend_v1.sql"

# Start snapshots are needed only until the run's last attempt ends (the
# outcome row keeps before/after for every prior the run changed), so they are
# dropped after this long. Durable retries land within hours.
SNAPSHOT_RETENTION_DAYS = 14.0

# asyncpg's UndefinedTableError. Matched on the SQLSTATE, not the message: a
# missing COLUMN also says "does not exist", and that is schema drift to
# report every time, not a migration to wait for. (Postgres also raises
# 42P01 for a query bug such as a missing FROM entry, so the one warning
# carries the error text.)
_UNDEFINED_TABLE = "42P01"

INSERT_DECISION_SQL = """
INSERT INTO curiosity_offer_decisions (
    run_id, arm, value_arm_propensity, offered, stale_offered, material_ids, constants
) VALUES ($1, $2, $3, $4::jsonb, $5::jsonb, $6::jsonb, $7::jsonb)
ON CONFLICT (run_id) DO NOTHING
"""

SET_TURN_SNAPSHOT_SQL = """
UPDATE curiosity_offer_decisions
SET turn_snapshot = $2::jsonb, turn_started_at = COALESCE(turn_started_at, now())
WHERE run_id = $1 AND turn_snapshot IS NULL
"""

PRUNE_TURN_SNAPSHOTS_SQL = """
UPDATE curiosity_offer_decisions
SET turn_snapshot = NULL
WHERE decided_at < now() - ($1 * interval '1 day') AND turn_snapshot IS NOT NULL
"""

LOAD_TURN_SNAPSHOT_SQL = """
SELECT turn_snapshot FROM curiosity_offer_decisions WHERE run_id = $1
"""

UPSERT_OUTCOME_SQL = """
INSERT INTO curiosity_run_outcomes (
    run_id, turn_ok, realized_nats, unknown_reason, n_tested, n_moved, n_formed,
    n_moved_untested, n_unattributed, n_invalid_confidence, per_prior, revision_agreement
) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11::jsonb, $12)
ON CONFLICT (run_id) DO UPDATE SET
    completed_at = now(),
    turn_ok = EXCLUDED.turn_ok,
    realized_nats = EXCLUDED.realized_nats,
    unknown_reason = EXCLUDED.unknown_reason,
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
_warned_no_pool = False


def _no_pool(what: str) -> None:
    """Hub has no memory Postgres pool: say so once, not never."""
    global _warned_no_pool
    if not _warned_no_pool:
        _warned_no_pool = True
        logger.warning(
            "curiosity_spend_log_no_pool op=%s -- Hub has no memory Postgres pool; "
            "offers and outcomes are not recorded and value ordering runs on a cold "
            "(uncertainty-equivalent) model",
            what,
        )


def _log_failure(what: str, run_id: Optional[str], exc: BaseException) -> None:
    global _warned_missing_table
    text = str(exc)
    if getattr(exc, "sqlstate", None) == _UNDEFINED_TABLE:
        if not _warned_missing_table:
            _warned_missing_table = True
            logger.warning(
                "curiosity_spend_log_table_missing op=%s err=%s -- apply %s; until then "
                "offers and outcomes are not recorded and value ordering runs on a "
                "cold (uncertainty-equivalent) model",
                what,
                text[:200],
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
    not answer -- an unreadable graph is not an empty one -- or when the
    answer filled the query's row cap: the cap has no order, so the start and
    end snapshots could hold different subsets and a missing prior would read
    as formed or unattributed."""
    try:
        rows = reader.query(ATLAS_PRIORS_CYPHER)
    except WorldviewUnavailable as exc:
        logger.warning("curiosity_spend_snapshot_unreadable err=%s", str(exc)[:200])
        return None
    if len(rows) >= ATLAS_PRIORS_LIMIT:
        logger.warning(
            "curiosity_spend_snapshot_truncated rows=%s limit=%s -- scored as unknown; "
            "raise ATLAS_PRIORS_LIMIT in orion/curiosity/atlas.py",
            len(rows),
            ATLAS_PRIORS_LIMIT,
        )
        return None
    # A forked prior (one id, several nodes) keeps the copy the OFFER keeps --
    # `Prior.fork_rank`, via `collapse_duplicate_priors`'s rule -- so the
    # "before" is the confidence Orion was shown, not a sibling's.
    best: dict[str, tuple[tuple, PriorState]] = {}
    for row in rows:
        state = PriorState.from_json(row)
        if state is None:
            continue
        prior = build_prior(row)
        rank = prior.fork_rank if prior is not None else (state.times_tested, "", "")
        seen = best.get(state.prior_id)
        if seen is None or rank > seen[0]:
            best[state.prior_id] = (rank, state)
    return {prior_id: state for prior_id, (_, state) in best.items()}


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
    it was offered at. Confidence is read with `valid_confidence`, the rule
    the outcome is scored with: a 1.7 or a NaN is recorded as no usable
    confidence (null), never as a clamped number -- and never as a NaN, which
    jsonb rejects, losing the whole row."""
    rows = []
    for rank, prior in enumerate(priors, start=1):
        confidence = valid_confidence(prior.confidence)
        rows.append(
            {
                "rank": rank,
                "prior_id": prior.prior_id,
                "confidence": confidence,
                "times_tested": prior.times_tested,
                "entropy_nats": entropy_nats(confidence),
                "yield": model.yield_for(prior.prior_id),
                "expected_nats": model.expected_nats(prior.prior_id, confidence),
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
    """One row per run, first write wins. Also drops start snapshots older
    than `SNAPSHOT_RETENTION_DAYS` -- once per run, so the table's largest
    column stays bounded without a separate job."""
    if pool is None:
        _no_pool("record_offer_decision")
        return False
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                INSERT_DECISION_SQL,
                run_id,
                arm,
                float(value_arm_propensity),
                json.dumps(offered, allow_nan=False),
                json.dumps(stale_offered, allow_nan=False),
                json.dumps(material_ids),
                json.dumps(dict(constants), allow_nan=False),
            )
    except Exception as exc:  # noqa: BLE001
        _log_failure("record_offer_decision", run_id, exc)
        return False
    try:
        async with pool.acquire() as conn:
            await conn.execute(PRUNE_TURN_SNAPSHOTS_SQL, SNAPSHOT_RETENTION_DAYS)
    except Exception as exc:  # noqa: BLE001
        _log_failure("prune_turn_snapshots", run_id, exc)
    return True


async def record_turn_snapshot(
    pool: Any, run_id: str, states: Optional[Mapping[str, PriorState]]
) -> bool:
    """The first snapshot taken wins (`turn_snapshot IS NULL`); the first
    attempt's start time is kept either way. `states=None` -- the graph could
    not be read -- marks the turn started with no snapshot, which a retry may
    fill; a filled-in start that already carries this run's stamps is scored
    unknown, never partial. A run with no decision row -- not an
    investigation, or its dispatch write failed -- is a no-op."""
    if pool is None:
        _no_pool("record_turn_snapshot")
        return False
    snapshot = (
        None
        if states is None
        else json.dumps([s.as_json() for s in states.values()], allow_nan=False)
    )
    try:
        async with pool.acquire() as conn:
            await conn.execute(SET_TURN_SNAPSHOT_SQL, run_id, snapshot)
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
        _no_pool("load_turn_snapshot")
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
    pool: Any,
    run_id: str,
    outcome: RunOutcome,
    agreement: Optional[float],
    *,
    turn_ok: bool,
) -> bool:
    """Upserted: a retry's end replaces an earlier attempt's, still scored
    from the first attempt's start. `turn_ok` is whether the turn produced
    text -- a failed turn that moved nothing reads 0.0 like a real
    "tested, nothing moved", and only this flag tells them apart."""
    if pool is None:
        _no_pool("record_run_outcome")
        return False
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                UPSERT_OUTCOME_SQL,
                run_id,
                bool(turn_ok),
                outcome.realized_nats,
                outcome.unknown_reason,
                outcome.n_tested,
                outcome.n_moved,
                outcome.n_formed,
                outcome.n_moved_untested,
                outcome.n_unattributed,
                outcome.n_invalid_confidence,
                json.dumps([o.as_json() for o in outcome.per_prior], allow_nan=False),
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
        _no_pool("load_prior_test_history")
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
