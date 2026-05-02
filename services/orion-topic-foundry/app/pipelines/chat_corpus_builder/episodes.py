from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime

from .types import AnchorRecord, ChatTurnRecord, EpisodeRecord, TurnGraphEdge


def segment_episodes(
    turns: list[ChatTurnRecord],
    anchors: list[AnchorRecord],
    edges: list[TurnGraphEdge],
    *,
    min_edge_weight: float = 1.0,
) -> list[EpisodeRecord]:
    turns_by_day: dict[str, list[ChatTurnRecord]] = defaultdict(list)
    for turn in turns:
        day = _iso_day(turn.created_at)
        turns_by_day[day].append(turn)
    anchors_by_turn = {item.turn_id: item.anchors for item in anchors}
    episodes: list[EpisodeRecord] = []
    for day in sorted(turns_by_day):
        day_turns = sorted(turns_by_day[day], key=lambda item: item.created_at)
        day_turn_ids = {item.turn_id for item in day_turns}
        day_edges = [
            edge
            for edge in edges
            if edge.weight >= min_edge_weight and edge.from_turn_id in day_turn_ids and edge.to_turn_id in day_turn_ids
        ]
        components = _connected_components(day_turns, day_edges)
        merged = _merge_tiny_components(components, anchors_by_turn)
        for idx, component in enumerate(merged, start=1):
            ordered = sorted(component, key=lambda item: item.created_at)
            top_anchors = _top_anchors(ordered, anchors_by_turn)
            label = _label_for_episode(top_anchors)
            confidence = min(0.99, 0.55 + (0.05 * min(8, len(top_anchors))) + (0.03 * min(10, len(ordered))))
            episodes.append(
                EpisodeRecord(
                    episode_id=f"chat-episode-{day.replace('-', '')}-{idx:03d}",
                    start_at=ordered[0].created_at,
                    end_at=ordered[-1].created_at,
                    turn_ids=[item.turn_id for item in ordered],
                    top_anchors=top_anchors,
                    confidence=round(confidence, 3),
                    episode_label=label,
                    episode_summary=f"{len(ordered)} turns linked by anchors: {', '.join(top_anchors[:4])}",
                )
            )
    return episodes


def _connected_components(turns: list[ChatTurnRecord], edges: list[TurnGraphEdge]) -> list[list[ChatTurnRecord]]:
    by_id = {item.turn_id: item for item in turns}
    adjacency: dict[str, set[str]] = {item.turn_id: set() for item in turns}
    for edge in edges:
        adjacency.setdefault(edge.from_turn_id, set()).add(edge.to_turn_id)
        adjacency.setdefault(edge.to_turn_id, set()).add(edge.from_turn_id)
    seen: set[str] = set()
    components: list[list[ChatTurnRecord]] = []
    for turn in turns:
        if turn.turn_id in seen:
            continue
        stack = [turn.turn_id]
        group_ids: list[str] = []
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            group_ids.append(node)
            stack.extend(adjacency.get(node, set()) - seen)
        components.append([by_id[item] for item in group_ids if item in by_id])
    return components


def _merge_tiny_components(
    components: list[list[ChatTurnRecord]],
    anchors_by_turn: dict[str, list[str]],
    min_size: int = 2,
) -> list[list[ChatTurnRecord]]:
    if len(components) <= 1:
        return components
    ordered = sorted(components, key=lambda group: min(item.created_at for item in group))
    merged: list[list[ChatTurnRecord]] = []
    for comp in ordered:
        if len(comp) >= min_size:
            merged.append(comp)
            continue
        if not merged:
            merged.append(comp)
            continue
        current_anchors = set(_flatten_anchors(comp, anchors_by_turn))
        previous_anchors = set(_flatten_anchors(merged[-1], anchors_by_turn))
        overlap = current_anchors.intersection(previous_anchors)
        if overlap:
            merged[-1].extend(comp)
        else:
            merged.append(comp)
    return merged


def _flatten_anchors(component: list[ChatTurnRecord], anchors_by_turn: dict[str, list[str]]) -> list[str]:
    out: list[str] = []
    for turn in component:
        out.extend(anchors_by_turn.get(turn.turn_id, []))
    return out


def _top_anchors(component: list[ChatTurnRecord], anchors_by_turn: dict[str, list[str]], top_k: int = 6) -> list[str]:
    counter: Counter[str] = Counter()
    for turn in component:
        counter.update(anchors_by_turn.get(turn.turn_id, []))
    return [key for key, _ in counter.most_common(top_k)]


def _label_for_episode(top_anchors: list[str]) -> str:
    if not top_anchors:
        return "chat turn cluster"
    return " / ".join(top_anchors[:3])


def _iso_day(value: str) -> str:
    return datetime.fromisoformat(value).date().isoformat()
