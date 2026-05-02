from __future__ import annotations

from collections import defaultdict
from datetime import datetime

from .types import AnchorRecord, ChatTurnRecord, GraphEdgeReason, TurnGraphEdge

CONFIRM_REPLIES = {"yup", "that worked", "fixed", "confirmed", "works"}
REJECT_REPLIES = {"still broken", "no", "didnt work", "didn't work", "fails"}


def build_turn_graph(turns: list[ChatTurnRecord], anchors: list[AnchorRecord], min_weight: float = 1.0) -> list[TurnGraphEdge]:
    turn_by_id = {turn.turn_id: turn for turn in turns}
    anchor_by_turn = {item.turn_id: set(item.anchors) for item in anchors}
    edge_map: dict[tuple[str, str], list[GraphEdgeReason]] = defaultdict(list)
    ordered = sorted(turns, key=lambda item: item.created_at)
    for idx, left in enumerate(ordered):
        for right in ordered[idx + 1 : min(idx + 9, len(ordered))]:
            reasons = _edge_reasons(left, right, anchor_by_turn.get(left.turn_id, set()), anchor_by_turn.get(right.turn_id, set()))
            weight = sum(item.reason_weight for item in reasons)
            if weight < min_weight:
                continue
            a, b = _ordered_pair(left.turn_id, right.turn_id)
            edge_map[(a, b)].extend(reasons)
    edges: list[TurnGraphEdge] = []
    for (from_turn_id, to_turn_id), reasons in edge_map.items():
        if from_turn_id not in turn_by_id or to_turn_id not in turn_by_id:
            continue
        edges.append(
            TurnGraphEdge(
                from_turn_id=from_turn_id,
                to_turn_id=to_turn_id,
                weight=round(sum(item.reason_weight for item in reasons), 4),
                reasons=reasons,
            )
        )
    return edges


def _edge_reasons(left: ChatTurnRecord, right: ChatTurnRecord, left_anchors: set[str], right_anchors: set[str]) -> list[GraphEdgeReason]:
    reasons: list[GraphEdgeReason] = []
    overlap = sorted(left_anchors.intersection(right_anchors))
    if overlap:
        reasons.append(GraphEdgeReason(reason_type="anchor_overlap", reason_weight=min(2.0, 0.35 * len(overlap)), detail=", ".join(overlap[:4])))
    delta_minutes = _minute_delta(left.created_at, right.created_at)
    if delta_minutes <= 12:
        reasons.append(GraphEdgeReason(reason_type="temporal_adjacent", reason_weight=0.6, detail=f"{delta_minutes}m"))
    left_text = f"{left.prompt}\n{left.response}".lower()
    right_text = f"{right.prompt}\n{right.response}".lower()
    if any(word in right_text for word in CONFIRM_REPLIES):
        reasons.append(GraphEdgeReason(reason_type="confirmation_signal", reason_weight=0.7))
    if any(word in right_text for word in REJECT_REPLIES):
        reasons.append(GraphEdgeReason(reason_type="rejection_signal", reason_weight=0.5))
    if left.has_commands and (right.has_logs or right.has_error):
        reasons.append(GraphEdgeReason(reason_type="command_result_link", reason_weight=0.8))
    if left.has_error and right.has_error and left_text == right_text:
        reasons.append(GraphEdgeReason(reason_type="same_error", reason_weight=0.8))
    return reasons


def _ordered_pair(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def _minute_delta(iso_left: str, iso_right: str) -> int:
    try:
        left = datetime.fromisoformat(iso_left)
        right = datetime.fromisoformat(iso_right)
    except Exception:
        return 99999
    return int(abs((right - left).total_seconds()) // 60)
