from __future__ import annotations

import math
from typing import Any, Optional

from orion.schemas.embodiment import WorldPerceptionV1


def build_perception(
    *, players: list[dict[str, Any]], orion_player_id: str, max_nearby: int = 8
) -> Optional[WorldPerceptionV1]:
    orion = next((p for p in players if str(p.get("id")) == orion_player_id), None)
    if orion is None or not isinstance(orion.get("position"), dict):
        return None
    ox, oy = float(orion["position"]["x"]), float(orion["position"]["y"])
    nearby = []
    for p in players:
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
    return WorldPerceptionV1(
        player_id=orion_player_id, position={"x": ox, "y": oy}, nearby_players=nearby[:max_nearby]
    )
