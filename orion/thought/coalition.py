from __future__ import annotations

from orion.schemas.thought import HubAssociationBundleV1, ThoughtEventV1

HUB_TURN_COALITION_PREFIX = "hub:turn:"
# What the stance prompt tells the model to write for this turn's anchor. The
# model never transcribes the correlation id: live 2026-10-02 (corr 39adc920)
# Qwen3.6-35B spent ~14k chars of reasoning re-checking a 36-char UUID copy,
# hit max_tokens with empty content, and the turn deferred.
HUB_TURN_REF_TOKEN = "hub:turn"


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


def canonicalize_turn_refs(refs: list[str], correlation_id: str) -> list[str]:
    """Expand any spelling of this turn's anchor to the real one, order kept, deduped.

    `hub:turn`, the bare correlation id, and any `hub:turn:<...>` (a stance pass
    only ever sees its own turn, so a mistyped id is this turn) all become
    `hub:turn:<correlation_id>`.
    """
    anchor = hub_turn_coalition_id(correlation_id)
    out: list[str] = []
    for ref in refs:
        if ref in (HUB_TURN_REF_TOKEN, correlation_id) or ref.startswith(HUB_TURN_COALITION_PREFIX):
            ref = anchor
        if ref not in out:
            out.append(ref)
    return out


def align_evidence_refs_to_coalition(
    thought: ThoughtEventV1,
    coalition_ids: set[str],
) -> ThoughtEventV1:
    """Snap LLM evidence_refs to coalition-backed ids; default to hub turn anchor.

    strain_refs are canonicalized too: they widen the allowed set, so a garbled
    anchor left there would launder the same garbled evidence ref.
    """
    strain_refs = canonicalize_turn_refs(thought.strain_refs, thought.correlation_id)
    allowed = coalition_ids | set(strain_refs)
    anchor = hub_turn_coalition_id(thought.correlation_id)
    valid = [
        ref
        for ref in canonicalize_turn_refs(thought.evidence_refs, thought.correlation_id)
        if ref in allowed
    ]
    if not valid and anchor in allowed:
        valid = [anchor]
    return thought.model_copy(update={"evidence_refs": valid, "strain_refs": strain_refs})
