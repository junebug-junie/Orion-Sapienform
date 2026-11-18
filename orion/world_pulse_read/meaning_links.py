"""List and withdraw the links one reading acceptance policy made (revert path).

``reading_meaning_target_v1`` (assertions.py) is Juniper's 2026-10-14 extension: it is
on by one flag and must be revertible by that flag plus this module. Turning the flag off
stops NEW acceptances; this module finds every assertion the policy (or any named
policy) accepted and appends a ``rejected`` decision on top, so Hub's AssertionProjector
closes the link on its next tick and the findings / link-seed readers (which read the
LATEST decision) drop it at once. The journal is append-only: nothing is deleted, the
withdrawal is itself an auditable decision with its own policy name.

Only a decision that is still the assertion's latest one is withdrawn. If something
later moved the assertion (Juniper reviewed it, another read decided it), that later
decision stands and the row is reported ``superseded``.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable, Optional

from orion.core.schemas.substrate_graph_journal import SubstrateGraphDecisionV1

from .assertions import MEANING_POLICY, READING_ACTOR

WITHDRAW_ACTOR = "reading_policy_withdraw"
WITHDRAWN_SUFFIX = "_withdrawn"
_NS = uuid.UUID("0f2a6c8e-3b1d-4e57-9a6c-71d2e5b4a903")
_TERMINAL = frozenset({"rejected", "deprecated"})

# Decisions by the named policies on reading proposals, each joined to its assertion's
# latest decision. $3 lower-bounds recorded_at; $4 keeps only decisions whose evidence
# names one of these reading seeds (``reading_seed:<id>``, set by plan_claims).
POLICY_DECISIONS_SQL = """
WITH mine AS (
    SELECT d.target_id, d.revision, d.payload
    FROM substrate_graph_journal d
    WHERE d.event_kind = 'decision' AND d.proposal_kind = 'relationship_assertion'
      AND d.payload->>'policy' = ANY($1::text[])
      AND EXISTS (SELECT 1 FROM substrate_graph_journal p
                  WHERE p.event_kind = 'proposal' AND p.proposal_id = d.proposal_id AND p.actor = $2)
      AND ($3::timestamptz IS NULL OR d.recorded_at >= $3)
      AND ($4::text[] IS NULL OR EXISTS (
          SELECT 1 FROM jsonb_array_elements_text(d.payload->'evidence_refs') r WHERE r = ANY($4::text[])))
), latest AS (
    SELECT DISTINCT ON (d.target_id) d.target_id, d.revision, d.payload
    FROM substrate_graph_journal d
    WHERE d.event_kind = 'decision' AND d.target_id IN (SELECT target_id FROM mine)
    ORDER BY d.target_id, d.revision DESC, d.recorded_at DESC
)
SELECT m.payload AS decided, l.revision AS latest_revision, l.payload AS latest
FROM mine m JOIN latest l USING (target_id)
ORDER BY m.target_id, m.revision
"""


def _payload(raw: Any) -> dict[str, Any]:
    return json.loads(raw) if isinstance(raw, str) else dict(raw)


@dataclass(frozen=True)
class PolicyLinkV1:
    decided: SubstrateGraphDecisionV1
    latest: SubstrateGraphDecisionV1
    latest_revision: int

    @property
    def status(self) -> str:
        """standing: still the latest decision, provisional/canonical; withdrawn: a later
        rejection/deprecation is already on top; superseded: something else decided later."""
        if self.latest.decision_id == self.decided.decision_id:
            return "withdrawn" if self.decided.resulting_state in _TERMINAL else "standing"
        if self.latest.resulting_state in _TERMINAL:
            return "withdrawn"
        return "superseded"

    def seed_ids(self) -> list[str]:
        return [r.split(":", 1)[1] for r in self.decided.evidence_refs if r.startswith("reading_seed:")]


def policy_links_from_rows(rows: Iterable[Any]) -> list[PolicyLinkV1]:
    return [PolicyLinkV1(decided=SubstrateGraphDecisionV1.model_validate(_payload(r["decided"])),
                         latest=SubstrateGraphDecisionV1.model_validate(_payload(r["latest"])),
                         latest_revision=int(r["latest_revision"]))
            for r in rows]


async def list_policy_links(
    conn: Any, *, policies: Iterable[str] = (MEANING_POLICY,), since: Optional[datetime] = None,
    seed_ids: Optional[Iterable[str]] = None,
) -> list[PolicyLinkV1]:
    seeds = [f"reading_seed:{s}" for s in seed_ids] if seed_ids is not None else None
    rows = await conn.fetch(POLICY_DECISIONS_SQL, list(policies), READING_ACTOR, since, seeds)
    return policy_links_from_rows(rows)


def withdrawal_for(link: PolicyLinkV1, *, recorded_at: datetime, reason: str = "") -> Optional[SubstrateGraphDecisionV1]:
    """The ``rejected`` decision that withdraws a standing link; None otherwise. Its id is
    derived from the decision it withdraws, so a re-run appends nothing twice."""
    if link.status != "standing":
        return None
    d = link.decided
    note = f": {reason}" if reason else ""
    return SubstrateGraphDecisionV1(
        proposal_id=d.proposal_id, proposal_kind=d.proposal_kind, target_id=d.target_id,
        actor=WITHDRAW_ACTOR, decision_id=f"reading-{uuid.uuid5(_NS, f'{d.decision_id}|withdraw')}",
        expected_prior_revision=link.latest_revision, resulting_state="rejected",
        policy=f"{d.policy}{WITHDRAWN_SUFFIX}", authority="local_inferred",
        rationale=f"withdrawn: link accepted by {d.policy} ({d.decision_id}){note}",
        evidence_refs=[f"withdraws:{d.decision_id}", *d.evidence_refs], recorded_at=recorded_at,
    )
