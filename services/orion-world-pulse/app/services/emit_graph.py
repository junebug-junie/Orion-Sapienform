from __future__ import annotations

from datetime import datetime, timezone

from orion.schemas.world_pulse import DailyWorldPulseV1, GraphDeltaPlanV1


def _escape_rdf_literal(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def build_graph_delta(
    digest: DailyWorldPulseV1,
    *,
    dry_run: bool,
    allowed_item_ids: set[str] | None = None,
) -> GraphDeltaPlanV1:
    triples = []
    for item in digest.items:
        if allowed_item_ids is not None and item.item_id not in allowed_item_ids:
            continue
        subj = f"<urn:orion:world-pulse:item:{item.item_id}>"
        triples.append(f"{subj} <http://conjourney.net/orion#category> \"{item.category}\" .")
        escaped_title = _escape_rdf_literal(item.title)
        triples.append(f"{subj} <http://conjourney.net/orion#title> \"{escaped_title}\" .")
        triples.append(f"{subj} <http://conjourney.net/orion#runId> \"{digest.run_id}\" .")
    return GraphDeltaPlanV1(
        graph_delta_id=f"graph:{digest.run_id}",
        run_id=digest.run_id,
        graph_name="http://conjourney.net/graph/world-pulse",
        triples="\n".join(triples),
        summary=f"{len(digest.items)} item deltas",
        triple_count=len(triples),
        policy_stamp={
            "graph_write_only_for_trust_tier_lte_3": True,
            "accepted_facts_blocked_for_low_trust": True,
        },
        dry_run=dry_run,
        created_at=datetime.now(timezone.utc),
    )
