from __future__ import annotations

from orion.schemas.thought import HubAssociationBundleV1


def coalition_ids_from_association(association: HubAssociationBundleV1) -> set[str]:
    """attended_node_ids + open_loop ids from broadcast frame."""
    ids: set[str] = set()
    broadcast = association.broadcast
    if broadcast is None:
        return ids
    ids.update(broadcast.attended_node_ids)
    for loop in broadcast.frame.open_loops:
        ids.add(loop.id)
    return ids
