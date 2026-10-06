"""Reader/writer for the append-only ``substrate_graph_journal`` table (asyncpg pool).

Contract: orion/core/schemas/substrate_graph_journal.py. Migration:
services/orion-sql-db/manual_migration_substrate_graph_journal_v1.sql.

Append is idempotent on event_id (a replayed producer writes nothing twice). A second
decision claiming the same next revision of a target raises ``RevisionConflict``
instead of silently winning: that is the expected-revision check #2497 requires.
There is no update or delete method; the journal is the recovery source for every
projection, so it only grows.
"""

from __future__ import annotations

import json
from typing import Any, Union

from orion.core.schemas.substrate_graph_journal import (
    SubstrateGraphDecisionV1,
    SubstrateGraphMaterializationV1,
    SubstrateGraphProposalV1,
)

JournalEventV1 = Union[SubstrateGraphProposalV1, SubstrateGraphDecisionV1, SubstrateGraphMaterializationV1]

_INSERT = """
INSERT INTO substrate_graph_journal (
    event_id, event_kind, proposal_kind, proposal_id, decision_id, target_id, revision, outcome,
    actor, payload, recorded_at
) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10::jsonb, $11)
ON CONFLICT (event_id) DO NOTHING
"""

# A decision is pending when it has no applied materialization, it is the NEXT revision of
# its target (or stale, so it can be refused once), and its last attempt did not fail for a
# terminal reason. Never-attempted decisions come first, then retries by oldest attempt, so a
# pile of decisions that keep failing (e.g. waiting for an endpoint) cannot starve new ones
# (#2515 review). Later revisions of a stuck target are not listed until it moves.
_PENDING_DECISIONS = """
WITH applied AS (
    SELECT target_id, MAX(revision) AS revision FROM substrate_graph_journal
    WHERE event_kind = 'materialization' AND outcome = 'applied' GROUP BY target_id),
last_attempt AS (
    SELECT DISTINCT ON (decision_id) decision_id, recorded_at, payload->>'failure_reason' AS reason
    FROM substrate_graph_journal WHERE event_kind = 'materialization'
    ORDER BY decision_id, recorded_at DESC, event_id DESC)
SELECT d.payload
FROM substrate_graph_journal d
LEFT JOIN applied a ON a.target_id = d.target_id
LEFT JOIN last_attempt m ON m.decision_id = d.decision_id
WHERE d.event_kind = 'decision'
  AND NOT EXISTS (
      SELECT 1 FROM substrate_graph_journal x
      WHERE x.event_kind = 'materialization' AND x.outcome = 'applied' AND x.decision_id = d.decision_id)
  AND d.revision - 1 <= COALESCE(a.revision, 0)
  AND NOT (m.reason IS NOT NULL AND split_part(m.reason, ':', 1) = ANY($2::text[]))
ORDER BY (m.decision_id IS NOT NULL), m.recorded_at NULLS FIRST, d.recorded_at, d.target_id, d.revision
LIMIT $1
"""

# Failure reasons that no retry can fix. Transient ones (an endpoint not written yet, a
# materializer error) are retried.
TERMINAL_FAILURE_REASONS: tuple[str, ...] = ("stale_revision", "proposal_missing_or_mismatched",
                                             "canonical_id_mismatch")

_PROPOSAL = """
SELECT payload FROM substrate_graph_journal WHERE event_kind = 'proposal' AND proposal_id = $1
"""

_LATEST_APPLIED_REVISION = """
SELECT COALESCE(MAX(revision), 0) AS revision FROM substrate_graph_journal
WHERE event_kind = 'materialization' AND outcome = 'applied' AND target_id = $1
"""

_FAILED_ATTEMPTS = """
SELECT count(*) AS n FROM substrate_graph_journal
WHERE event_kind = 'materialization' AND outcome = 'failed' AND decision_id = $1
"""

_LAST_MATERIALIZATION = """
SELECT payload FROM substrate_graph_journal
WHERE event_kind = 'materialization' AND decision_id = $1
ORDER BY recorded_at DESC, event_id DESC LIMIT 1
"""



class RevisionConflict(RuntimeError):
    """Another decision already claimed this target's next revision."""


def _payload(row: Any) -> Any:
    raw = row["payload"]
    return json.loads(raw) if isinstance(raw, str) else raw


def journal_row(event: JournalEventV1) -> tuple[Any, ...]:
    """Column values for one event, in _INSERT's order (also used by non-asyncpg writers)."""
    decision_id = getattr(event, "decision_id", None)
    if isinstance(event, SubstrateGraphDecisionV1):
        revision, outcome = event.resulting_revision, None
    elif isinstance(event, SubstrateGraphMaterializationV1):
        revision, outcome = event.revision, event.outcome
    else:
        revision, outcome = None, None
    return (
        # Namespaced by kind: a proposal id and a decision id can never collide.
        f"{event.event_kind}:{event.event_id}", event.event_kind, event.proposal_kind, event.proposal_id, decision_id,
        event.target_id, revision, outcome, event.actor, event.model_dump_json(), event.recorded_at,
    )


class SubstrateGraphJournal:
    def __init__(self, pool: Any) -> None:
        self._pool = pool

    async def append(self, event: JournalEventV1) -> bool:
        """Insert one event. Returns False when the same event_id was already recorded."""
        import asyncpg

        try:
            async with self._pool.acquire() as conn:
                status = await conn.execute(_INSERT, *journal_row(event))
        except asyncpg.UniqueViolationError as exc:
            raise RevisionConflict(
                f"target {event.target_id} already has a decision at revision "
                f"{getattr(event, 'resulting_revision', '?')}"
            ) from exc
        return status.endswith(" 1")

    async def pending_decisions(self, *, limit: int = 100) -> list[SubstrateGraphDecisionV1]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(_PENDING_DECISIONS, int(limit), list(TERMINAL_FAILURE_REASONS))
        return [SubstrateGraphDecisionV1.model_validate(_payload(row)) for row in rows]

    async def proposal(self, proposal_id: str) -> SubstrateGraphProposalV1 | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(_PROPOSAL, proposal_id)
        return SubstrateGraphProposalV1.model_validate(_payload(row)) if row else None

    async def latest_applied_revision(self, target_id: str) -> int:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(_LATEST_APPLIED_REVISION, target_id)
        return int(row["revision"])

    async def failed_attempts(self, decision_id: str) -> int:
        async with self._pool.acquire() as conn:
            return int(await conn.fetchval(_FAILED_ATTEMPTS, decision_id))

    async def last_materialization(self, decision_id: str) -> SubstrateGraphMaterializationV1 | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(_LAST_MATERIALIZATION, decision_id)
        return SubstrateGraphMaterializationV1.model_validate(_payload(row)) if row else None
