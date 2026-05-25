"""Node-scoped filtering for substrate receipts and organ emissions."""

from __future__ import annotations

from typing import Any

from orion.schemas.organ_emission import OrganEmissionV1
from orion.schemas.reduction_receipt import ReductionReceiptV1


def _normalize_node_id(node_id: str) -> str:
    return node_id.strip().lower()


def receipt_touches_node(receipt: ReductionReceiptV1, node_id: str) -> bool:
    nid = _normalize_node_id(node_id)
    for delta in receipt.state_deltas:
        if _normalize_node_id(delta.target_id) == nid:
            return True
    for update in receipt.projection_updates:
        if update.node_id and _normalize_node_id(update.node_id) == nid:
            return True
    return False


def emission_touches_node(emission: OrganEmissionV1, node_id: str) -> bool:
    nid = _normalize_node_id(node_id)
    needle = f":{nid}:"
    for event in emission.candidate_events:
        if event.trace_id and needle in event.trace_id:
            return True
        if event.atom and event.atom.text_value:
            if _normalize_node_id(event.atom.text_value) == nid:
                return True
    return False


def state_deltas_for_node(receipt: ReductionReceiptV1, node_id: str) -> list[dict[str, Any]]:
    nid = _normalize_node_id(node_id)
    return [
        delta.model_dump(mode="json")
        for delta in receipt.state_deltas
        if _normalize_node_id(delta.target_id) == nid
    ]
