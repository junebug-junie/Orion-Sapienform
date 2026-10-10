"""Link-accepted curiosity seeds: an accepted reading link becomes one curiosity seed.

Juniper's decision 2 (2026-10-10). When a reading claim is accepted (journal decision by
``reading_quote_rule_v1`` on a proposal from ``world_pulse_read_stage2``, state
provisional/canonical) AND its projection was applied, the curiosity tick mints one
seed whose focal nodes are the claim's two endpoints. The accepted link then sits
BETWEEN two focal nodes, so the neighborhood attach step
(orion/substrate/curiosity_seed_neighborhood.py) reports it in ``focal_edge_refs``.

This is an event, not a score. The seed carries ``signal_strength=0.0`` and
``confidence=0.0`` with the note ``strength:unscored_event``: it never outranks a scored
seed in the frontier decision or in the chat-stance top-3, and it never reaches the
evaluator's invoke threshold (0.5) on its own. No metric is introduced.

Guardrails:
- once per assertion revision: the seed carries ``link_assertion:<id>@<rev>``; a link
  already present in a stored candidate set is never minted again (the SQL below);
- only the LATEST decision per assertion counts, so a later rejected/deprecated decision
  stops the seed even if an older accepted projection was applied;
- per-tick cap (``HARD_LINK_SEED_CEILING``), oldest first, appended after the scored
  seeds and stored in addition to them, so they never displace one.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable, Mapping

from orion.core.schemas.frontier_curiosity import FrontierInvocationSignalV1
from orion.substrate.assertion_projector import _edge_id as _projector_edge_id
from orion.substrate.neighborhood import ACCEPTED_ASSERTION_STATES

# orion/world_pulse_read/assertions.py READING_ACTOR (not imported: that module pulls the
# whole reading stack into substrate-runtime). A test pins the two equal.
READING_PROPOSAL_ACTORS: tuple[str, ...] = ("world_pulse_read_stage2",)
SOURCE_NOTE = "source:reading_link_accepted"
UNSCORED_NOTE = "strength:unscored_event"
HARD_LINK_SEED_CEILING = 4
# The idempotency check looks for an existing seed in candidate sets stored since the
# projection was applied, minus this slack for clock skew between the Hub (which stamps
# recorded_at) and substrate-runtime (which stamps generated_at).
CLOCK_SKEW_SLACK_HOURS = 1.0


def link_seed_key(assertion_id: str, revision: int) -> str:
    return f"link_assertion:{assertion_id}@{int(revision)}"


def projection_edge_id(assertion_id: str) -> str:
    """The id AssertionProjector gives an assertion's semantic_projection edge."""
    return _projector_edge_id(assertion_id, "semantic_projection")


# One round trip. Latest decision per assertion -> must be accepted, applied, proposed by
# a reading actor, applied within the lookback, and not already in a stored seed.
ACCEPTED_UNSEEDED_LINKS_SQL = """
WITH latest AS (
    SELECT DISTINCT ON (d.target_id)
           d.target_id, d.revision, d.decision_id, d.proposal_id,
           d.payload->>'resulting_state' AS state
    FROM substrate_graph_journal d
    WHERE d.event_kind = 'decision' AND d.proposal_kind = 'relationship_assertion'
    ORDER BY d.target_id, d.revision DESC, d.recorded_at DESC
)
SELECT l.target_id AS assertion_id,
       l.revision AS revision,
       l.state AS state,
       p.payload->>'subject_node_id' AS subject_node_id,
       p.payload->>'object_node_id' AS object_node_id,
       p.payload->>'predicate' AS predicate,
       p.payload->>'statement_text' AS statement_text,
       m.payload->'edge_ids' AS edge_ids,
       m.recorded_at AS materialized_at
FROM latest l
JOIN substrate_graph_journal m
  ON m.event_kind = 'materialization' AND m.outcome = 'applied' AND m.decision_id = l.decision_id
JOIN substrate_graph_journal p
  ON p.event_kind = 'proposal' AND p.proposal_id = l.proposal_id
WHERE p.actor = ANY(:actors)
  AND l.state = ANY(:accepted_states)
  AND m.recorded_at >= now() - (:lookback_hours * interval '1 hour')
  AND NOT EXISTS (
      SELECT 1
      FROM substrate_endogenous_curiosity_candidates c
      CROSS JOIN LATERAL jsonb_array_elements(c.candidates_json) s
      WHERE c.generated_at >= m.recorded_at - (:skew_hours * interval '1 hour')
        AND s->'notes' @> jsonb_build_array('link_assertion:' || l.target_id || '@' || l.revision::text)
  )
ORDER BY m.recorded_at, l.target_id
LIMIT :limit
"""


def accepted_links_params(*, lookback_hours: float, limit: int) -> dict[str, Any]:
    return {
        "actors": list(READING_PROPOSAL_ACTORS),
        "accepted_states": list(ACCEPTED_ASSERTION_STATES),
        "lookback_hours": float(lookback_hours),
        "skew_hours": CLOCK_SKEW_SLACK_HOURS,
        "limit": int(limit),
    }


@dataclass(frozen=True)
class AcceptedLinkV1:
    assertion_id: str
    revision: int
    subject_node_id: str
    object_node_id: str
    predicate: str
    statement_text: str
    projection_edge_id: str
    materialized_at: datetime | None = None


def links_from_rows(rows: Iterable[Mapping[str, Any]]) -> tuple[list[AcceptedLinkV1], int]:
    """Rows of ACCEPTED_UNSEEDED_LINKS_SQL -> links whose projection edge was written.

    Returns (links, skipped): a row whose applied materialization does not list the
    projection edge (should not happen for an accepted decision) is skipped, not seeded.
    """
    links: list[AcceptedLinkV1] = []
    skipped = 0
    for row in rows:
        assertion_id = str(row.get("assertion_id") or "")
        subject = str(row.get("subject_node_id") or "")
        obj = str(row.get("object_node_id") or "")
        if not assertion_id or not subject or not obj or subject == obj:
            skipped += 1
            continue
        edge_id = projection_edge_id(assertion_id)
        if edge_id not in [str(e) for e in (row.get("edge_ids") or [])]:
            skipped += 1
            continue
        links.append(
            AcceptedLinkV1(
                assertion_id=assertion_id,
                revision=int(row.get("revision") or 0),
                subject_node_id=subject,
                object_node_id=obj,
                predicate=str(row.get("predicate") or ""),
                statement_text=str(row.get("statement_text") or ""),
                projection_edge_id=edge_id,
                materialized_at=row.get("materialized_at"),
            )
        )
    return links, skipped


def link_accepted_seed(
    link: AcceptedLinkV1, *, anchor_scope: str = "orion", subject_ref: str | None = "entity:orion"
) -> FrontierInvocationSignalV1:
    statement = " ".join(link.statement_text.split())[:240]
    return FrontierInvocationSignalV1(
        signal_type="curiosity_candidate",
        anchor_scope=anchor_scope,
        subject_ref=subject_ref,
        target_zone="concept_graph",
        task_type_candidate="relation_discovery",
        focal_node_refs=[link.subject_node_id, link.object_node_id],
        signal_strength=0.0,
        confidence=0.0,
        evidence_summary=(
            f"a reading link was accepted: {link.subject_node_id} {link.predicate} "
            f"{link.object_node_id}" + (f" ({statement})" if statement else "")
        ),
        notes=[
            "endogenous_seed",
            SOURCE_NOTE,
            UNSCORED_NOTE,
            link_seed_key(link.assertion_id, link.revision),
            f"projection_edge:{link.projection_edge_id}",
        ],
    )


def link_accepted_seeds(
    links: Iterable[AcceptedLinkV1], *, cap: int, anchor_scope: str = "orion",
    subject_ref: str | None = "entity:orion",
) -> list[FrontierInvocationSignalV1]:
    bounded = max(0, min(int(cap), HARD_LINK_SEED_CEILING))
    out: list[FrontierInvocationSignalV1] = []
    seen: set[str] = set()
    for link in links:
        key = link_seed_key(link.assertion_id, link.revision)
        if key in seen:
            continue
        seen.add(key)
        out.append(link_accepted_seed(link, anchor_scope=anchor_scope, subject_ref=subject_ref))
        if len(out) >= bounded:
            break
    return out
