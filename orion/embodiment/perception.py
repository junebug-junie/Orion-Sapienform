from __future__ import annotations

import math
from typing import Any, Optional

from orion.schemas.embodiment import WorldPerceptionV1


def _active_conversation(
    conversations: list[dict[str, Any]],
    orion_player_id: str,
    players_by_id: dict[str, dict[str, Any]],
    messages: list[dict[str, Any]],
) -> Optional[dict[str, Any]]:
    """Shape the conversation Orion is a member of (invited/walkingOver/participating).

    ``participants`` is a flat id list so ``should_speak`` can test membership;
    ``status`` lets the worker gate accept/walk-over/speak; ``other`` carries the
    partner's name+position for walk-over; ``messages`` feeds the speech prompt.
    """
    for cv in conversations or []:
        parts = cv.get("participants") or []
        ids = [str(p.get("playerId")) for p in parts if p.get("playerId") is not None]
        if orion_player_id not in ids:
            continue
        status = next(
            ((p.get("status") or {}).get("kind") for p in parts
             if str(p.get("playerId")) == orion_player_id),
            None,
        )
        others = [i for i in ids if i != orion_player_id]
        other = None
        if others:
            op = players_by_id.get(others[0], {})
            other = {"player_id": others[0], "name": op.get("name"), "position": op.get("position")}
        shaped_msgs = [
            {
                "author_id": str(m.get("author")) if m.get("author") is not None else None,
                "author": m.get("authorName") or m.get("author"),
                "text": str(m.get("text") or "").strip(),
            }
            for m in (messages or [])
            if str(m.get("text") or "").strip()
        ]
        return {
            "conversation_id": str(cv.get("id")),
            "id": str(cv.get("id")),
            "status": status,
            "participants": ids,
            "other": other,
            "messages": shaped_msgs,
        }
    return None


def build_perception(
    *,
    players: list[dict[str, Any]],
    orion_player_id: str,
    conversations: Optional[list[dict[str, Any]]] = None,
    messages: Optional[list[dict[str, Any]]] = None,
    max_nearby: int = 8,
) -> Optional[WorldPerceptionV1]:
    orion = next((p for p in players if str(p.get("id")) == orion_player_id), None)
    if orion is None or not isinstance(orion.get("position"), dict):
        return None
    ox, oy = float(orion["position"]["x"]), float(orion["position"]["y"])
    nearby = []
    players_by_id: dict[str, dict[str, Any]] = {}
    for p in players:
        players_by_id[str(p.get("id"))] = p
        if str(p.get("id")) == orion_player_id or not isinstance(p.get("position"), dict):
            continue
        px, py = float(p["position"]["x"]), float(p["position"]["y"])
        nearby.append({
            "player_id": str(p.get("id")),
            "name": p.get("name"),
            "position": {"x": px, "y": py},
            "distance": round(math.hypot(px - ox, py - oy), 4),
        })
    nearby.sort(key=lambda n: n["distance"])
    active = _active_conversation(conversations or [], orion_player_id, players_by_id, messages or [])
    return WorldPerceptionV1(
        player_id=orion_player_id, position={"x": ox, "y": oy},
        nearby_players=nearby[:max_nearby], active_conversation=active,
    )
