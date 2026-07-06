from __future__ import annotations

from orion.schemas.thought import HubAssociationBundleV1, ThoughtEventV1

HUB_TURN_COALITION_PREFIX = "hub:turn:"


def hub_turn_coalition_id(correlation_id: str) -> str:
    return f"{HUB_TURN_COALITION_PREFIX}{correlation_id}"


def coalition_ids_from_association(association: HubAssociationBundleV1) -> set[str]:
    """attended_node_ids + open_loop ids + always the current Hub turn anchor."""
    ids: set[str] = {hub_turn_coalition_id(association.correlation_id)}
    broadcast = association.broadcast
    if broadcast is None:
        return ids
    ids.update(broadcast.attended_node_ids)
    for loop in broadcast.frame.open_loops:
        ids.add(loop.id)
    return ids


def align_evidence_refs_to_coalition(
    thought: ThoughtEventV1,
    coalition_ids: set[str],
) -> ThoughtEventV1:
    """Snap LLM evidence_refs to coalition-backed ids; default to hub turn anchor."""
    allowed = coalition_ids | set(thought.strain_refs)
    anchor = hub_turn_coalition_id(thought.correlation_id)
    valid: list[str] = []
    for ref in thought.evidence_refs:
        if ref in allowed:
            valid.append(ref)
        elif ref == thought.correlation_id and anchor in allowed:
            valid.append(anchor)
    if not valid and anchor in allowed:
        valid = [anchor]
    return thought.model_copy(update={"evidence_refs": valid})
