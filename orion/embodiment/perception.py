from __future__ import annotations

import math
from typing import Any, Optional

from orion.schemas.embodiment import WorldPerceptionV1


def _facing_partner(
    orion_facing: Optional[dict[str, Any]],
    orion_position: Optional[dict[str, Any]],
    partner_position: Optional[dict[str, Any]],
    *,
    tolerance: float = 0.7,
) -> Optional[bool]:
    """True iff Orion's facing vector aligns with the direction to the partner.

    Compares the normalized ``facing`` ({dx,dy}) against the normalized vector
    from Orion to the partner via dot product; aligned when the dot product is
    at least ``tolerance``. Returns ``None`` when facing/positions are unknown so
    callers can distinguish "not facing" from "unknown".
    """
    if not isinstance(orion_facing, dict) or not isinstance(orion_position, dict):
        return None
    if not isinstance(partner_position, dict):
        return None
    try:
        fx, fy = float(orion_facing.get("dx")), float(orion_facing.get("dy"))
        ox, oy = float(orion_position["x"]), float(orion_position["y"])
        px, py = float(partner_position["x"]), float(partner_position["y"])
    except (TypeError, ValueError, KeyError):
        return None
    fmag = math.hypot(fx, fy)
    to_dx, to_dy = px - ox, py - oy
    tmag = math.hypot(to_dx, to_dy)
    if fmag == 0.0 or tmag == 0.0:
        return None
    dot = (fx / fmag) * (to_dx / tmag) + (fy / fmag) * (to_dy / tmag)
    return dot >= tolerance


def _active_conversation(
    conversations: list[dict[str, Any]],
    orion_player_id: str,
    players_by_id: dict[str, dict[str, Any]],
    messages: list[dict[str, Any]],
    orion_facing: Optional[dict[str, Any]] = None,
    orion_position: Optional[dict[str, Any]] = None,
) -> Optional[dict[str, Any]]:
    """Shape the conversation Orion is a member of (invited/walkingOver/participating).

    ``participants`` is a flat id list so ``should_speak`` can test membership;
    ``status`` lets the worker gate accept/walk-over/speak; ``other`` carries the
    partner's name+position for walk-over; ``messages`` feeds the speech prompt.
    When ``participating``, ``facing_partner`` reports whether Orion is oriented
    toward the partner (None if facing/positions are unknown).
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
            other = {
                "player_id": others[0],
                "name": op.get("name"),
                "position": op.get("position"),
                # ai-town's raw player record carries `human` (a session-token id)
                # only for human-controlled players; absent for ai-town-agent NPCs.
                # Forwarded (previously dropped here) so callers can tag
                # conversation-memory events with the correct participant_kind.
                "is_human": bool(op.get("human")),
            }
        shaped_msgs = [
            {
                "author_id": str(m.get("author")) if m.get("author") is not None else None,
                "author": m.get("authorName") or m.get("author"),
                "text": str(m.get("text") or "").strip(),
            }
            for m in (messages or [])
            if str(m.get("text") or "").strip()
        ]
        facing_partner: Optional[bool] = None
        if status == "participating" and other is not None:
            facing_partner = _facing_partner(
                orion_facing, orion_position, other.get("position")
            )
        return {
            "conversation_id": str(cv.get("id")),
            "id": str(cv.get("id")),
            "status": status,
            "participants": ids,
            "other": other,
            "messages": shaped_msgs,
            "facing_partner": facing_partner,
        }
    return None


def _bresenham_tiles(x0: int, y0: int, x1: int, y1: int) -> list[tuple[int, int]]:
    """Integer grid tiles on the line from (x0,y0) to (x1,y1), inclusive of both ends."""
    tiles = []
    dx, dy = abs(x1 - x0), -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    x, y = x0, y0
    while True:
        tiles.append((x, y))
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy
    return tiles


def _has_line_of_sight(
    walkable: Optional[set[tuple[int, int]]], ox: float, oy: float, tx: float, ty: float,
) -> bool:
    """Approximate line-of-sight: true unless a non-walkable tile lies strictly
    between Orion and the target.

    ``walkable`` is AI Town's own movement-collision layer (orion/embodiment/
    worldmap.py's walkable_tiles), reused here as a stand-in for sight-blocking
    -- an honest approximation, not a real visibility layer: the map has no
    distinct "blocks sight" data, so a walk-blocking-but-see-through tile (e.g.
    a short fence) would be wrongly treated as blocking, and the reverse
    (sight-blocking-but-walkable) can't be represented at all. Endpoint tiles
    (Orion's own tile and the target's, which may itself be a solid,
    non-walkable structure like a windmill) are never checked -- you can see a
    windmill even though you can't walk onto its tile. Fail-open (returns True)
    when no map data is available, so a missing/failed map load never hides
    something that's actually there.
    """
    if not walkable:
        return True
    x0, y0 = round(ox), round(oy)
    x1, y1 = round(tx), round(ty)
    for x, y in _bresenham_tiles(x0, y0, x1, y1):
        if (x, y) in ((x0, y0), (x1, y1)):
            continue
        if (x, y) not in walkable:
            return False
    return True


def _nearby_landmarks(
    locations: dict[str, Any], ox: float, oy: float, *, max_landmarks: int,
    walkable: Optional[set[tuple[int, int]]] = None,
) -> list[dict[str, Any]]:
    """Named static map features near (ox, oy), nearest first.

    ``locations`` is the same name -> {x, y} registry ``go_to_location``
    already uses for movement targeting (EMBODIMENT_LOCATIONS_JSON) -- reused
    here read-only so Orion's perception (and therefore speech) can reference
    real places on the map, not just other players.
    """
    landmarks = []
    for name, loc in (locations or {}).items():
        if not isinstance(loc, dict):
            continue
        try:
            lx, ly = float(loc["x"]), float(loc["y"])
        except (TypeError, ValueError, KeyError):
            continue
        landmarks.append({
            "name": str(name),
            "position": {"x": lx, "y": ly},
            "distance": round(math.hypot(lx - ox, ly - oy), 4),
            "is_visible": _has_line_of_sight(walkable, ox, oy, lx, ly),
        })
    landmarks.sort(key=lambda n: n["distance"])
    return landmarks[:max_landmarks]


def build_perception(
    *,
    players: list[dict[str, Any]],
    orion_player_id: str,
    conversations: Optional[list[dict[str, Any]]] = None,
    messages: Optional[list[dict[str, Any]]] = None,
    locations: Optional[dict[str, Any]] = None,
    walkable: Optional[set[tuple[int, int]]] = None,
    max_nearby: int = 8,
    max_landmarks: int = 3,
) -> Optional[WorldPerceptionV1]:
    orion = next((p for p in players if str(p.get("id")) == orion_player_id), None)
    if orion is None or not isinstance(orion.get("position"), dict):
        return None
    ox, oy = float(orion["position"]["x"]), float(orion["position"]["y"])
    # Own orientation + movement state (raw serialized-player fields; see
    # convex/aiTown/player.ts serialize()). `pathfinding` is present/truthy only
    # while the engine is moving Orion along a path.
    orion_facing = orion.get("facing") if isinstance(orion.get("facing"), dict) else None
    orion_pathfinding = bool(orion.get("pathfinding"))
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
            "is_human": bool(p.get("human")),
            "is_visible": _has_line_of_sight(walkable, ox, oy, px, py),
        })
    nearby.sort(key=lambda n: n["distance"])
    orion_position = {"x": ox, "y": oy}
    active = _active_conversation(
        conversations or [], orion_player_id, players_by_id, messages or [],
        orion_facing=orion_facing, orion_position=orion_position,
    )
    landmarks = _nearby_landmarks(locations or {}, ox, oy, max_landmarks=max_landmarks, walkable=walkable)
    return WorldPerceptionV1(
        player_id=orion_player_id, position=orion_position,
        facing=orion_facing, pathfinding=orion_pathfinding,
        nearby_players=nearby[:max_nearby], nearby_landmarks=landmarks,
        active_conversation=active,
    )
